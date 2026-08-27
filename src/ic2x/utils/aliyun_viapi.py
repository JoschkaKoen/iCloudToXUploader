"""
Alibaba Cloud VIAPI (imageenhan) image enhancement — super-resolution + color.

Auth is the Aliyun **AccessKey** (ALIBABA_CLOUD_ACCESS_KEY_ID / _SECRET), distinct
from the DashScope bearer key the Qwen judges use. The SDK's ``*_advance`` methods
auto-upload a local file to a temporary OSS bucket (no OSS setup), call the API
synchronously, and return a result-image URL valid ~30 min, which we download here.

Capabilities (UpscaleFactor=1 → enhance-only, no upscaling):
  superres  MakeSuperResolutionImage  denoise/sharpen/detail   (≤5 MB, ≤1920×1080)
  color     EnhanceImageColor         auto saturation/brightness/contrast
                                       (<3 MB, 64×64..3840×2160; mode Rec709|ln17_256|LogC)

Each capability must be activated once in the VIAPI console (imageenhan category) or
calls return InvalidApi.NotPurchase. First 100 calls/month free, then 0.02 RMB; QPS 2.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger("ic2x.viapi")

ENDPOINT = "imageenhan.cn-shanghai.aliyuncs.com"

# Enhancement fails OPEN (the original posts), so a blip costs a post its colour work
# silently. Measured 2026-08-04..09: 4 of 18 posts went out unenhanced, on four
# different transient faults — read timeout, two SSL EOFs, and a truncated result
# download. Retry in close succession like every other network call in a cycle.
_ENHANCE_ATTEMPTS = 3
_ENHANCE_RETRY_DELAY = 3.0
# The result download is retried on its own INSIDE each attempt: the OSS URL stays
# valid, so re-fetching costs nothing while re-running the API call costs a quota call.
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_RETRY_DELAY = 2.0

# Input limits that satisfy BOTH super-res (≤1920×1080, ≤5 MB) and color
# (≤3840×2160 long / ≤2160 short, <3 MB): fit inside 1920×1080 and keep <3 MB.
MAX_LONG_EDGE = 1920
MAX_SHORT_EDGE = 1080
MAX_BYTES = 3_000_000

COLOR_MODES = ("Rec709", "ln17_256", "LogC")  # Rec709 = natural, ln17_256 = stronger


class ViapiError(RuntimeError):
    """A VIAPI call could not complete (missing creds, not activated, bad input, network)."""


def check_credentials() -> tuple[str, str]:
    """Return (access_key_id, access_key_secret) or raise a clear ViapiError."""
    ak = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "").strip()
    sk = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "").strip()
    if not ak or not sk:
        raise ViapiError(
            "Aliyun AccessKey missing — set ALIBABA_CLOUD_ACCESS_KEY_ID and "
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET in .env (VIAPI uses the AccessKey, NOT "
            "the DashScope key).")
    return ak, sk


def _client():
    ak, sk = check_credentials()
    from alibabacloud_imageenhan20190930.client import Client
    from alibabacloud_tea_openapi.models import Config
    return Client(Config(access_key_id=ak, access_key_secret=sk, endpoint=ENDPOINT))


def _run(method: str, build_request, url_attr: str, image_path: Path,
         _attempt: int = 1) -> bytes:
    """Open the file, call the *_advance method (auto-OSS-upload, synchronous),
    download the result image. Raises ViapiError with a clean message on any failure.

    Retried in close succession on TRANSIENT failures. Enhancement fails open — the
    original photo posts — so every blip silently costs a post its colour work, and
    that was happening to 4 of 18 posts (22%) over 2026-08-04..09. The SDK's own
    `autoretry` does not cover the result download, which we issue ourselves, and one
    of the four failures was exactly that: `result download failed: IncompleteRead
    (23865 bytes read, 622255 more expected)` — the enhancement had already succeeded
    server-side. A permanent error (capability not activated, bad credentials) is
    raised immediately; retrying it would just delay every post."""
    from alibabacloud_tea_util.models import RuntimeOptions

    # The *_advance call uploads the file to a temp OSS bucket first; multi-MB photos
    # exceed the 10s default, so widen the timeouts (ms) and let it retry once.
    #
    # connect_timeout must be generous, not just read_timeout: the SDK reports its
    # read timeouts using the CONNECT value ("Read timed out. (read timeout=15.0)"
    # while read_timeout was 90000), so connect_timeout is the limit that actually
    # binds. At 15s it was too tight for a cold call — measured on the render-worker
    # host 2026-08-04, a cold enhance took 18.5s and a warm one 5.6s, so the first
    # enhance after startup timed out every time and the photo posted unenhanced.
    runtime = RuntimeOptions(connect_timeout=60000, read_timeout=90000,
                             autoretry=True, max_attempts=2)
    client = _client()

    def _retry_or_raise(err: "ViapiError") -> bytes:
        if _attempt >= _ENHANCE_ATTEMPTS:
            raise err
        logger.warning("viapi: %s failed (attempt %d/%d), retrying: %s",
                       method, _attempt, _ENHANCE_ATTEMPTS, str(err)[:160])
        time.sleep(_ENHANCE_RETRY_DELAY)
        return _run(method, build_request, url_attr, image_path, _attempt + 1)

    try:
        with open(image_path, "rb") as f:
            resp = getattr(client, method)(build_request(f), runtime)
        url = getattr(resp.body.data, url_attr, None)
    except ViapiError:
        raise
    except Exception as exc:  # noqa: BLE001 — collapse SDK/network errors to one clean type
        msg = str(exc)
        if "NotPurchase" in msg:
            # Permanent: the capability is not switched on. Retrying cannot help.
            raise ViapiError(
                f"{method}: imageenhan capability not activated for this account — open "
                "it at https://common-buy.aliyun.com/?commodityCode=viapi_imageenhan_"
                "public_cn (first 100 calls/month free).") from exc
        return _retry_or_raise(ViapiError(f"{method}: {msg}"))
    if not url:
        return _retry_or_raise(ViapiError(f"{method}: API returned no result URL"))
    # The OSS result URL comes back as plain http, which the GFW resets routinely —
    # 2026-08-19 every attempt died on `HTTPConnectionPool(... port=80)` with truncated
    # reads, so the enhancement succeeded server-side and we threw it away. Force https,
    # and retry the DOWNLOAD by itself: the URL stays valid, so re-fetching is free
    # whereas re-running the whole API call is not.
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    last: Exception | None = None
    for dl_try in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if dl_try < _DOWNLOAD_ATTEMPTS:
                logger.warning("viapi: result download failed (%d/%d), retrying: %s",
                               dl_try, _DOWNLOAD_ATTEMPTS, str(exc)[:120])
                time.sleep(_DOWNLOAD_RETRY_DELAY)
    return _retry_or_raise(ViapiError(f"{method}: result download failed: {last}"))


def superres(image_path: Path, *, upscale: int = 1, quality: int = 95) -> bytes:
    """Super-resolution enhance (upscale=1 → enhance-only, no enlarging). Returns JPEG bytes."""
    import alibabacloud_imageenhan20190930.models as m
    return _run(
        "make_super_resolution_image_advance",
        lambda f: m.MakeSuperResolutionImageAdvanceRequest(
            url_object=f, upscale_factor=upscale, output_format="jpg", output_quality=quality),
        "url", image_path)


def color_enhance(image_path: Path, *, mode: str = "Rec709") -> bytes:
    """Auto color enhancement (saturation/brightness/contrast). Returns JPEG bytes.
    mode: Rec709 (natural) | ln17_256 (stronger) | LogC (low-contrast/film input)."""
    import alibabacloud_imageenhan20190930.models as m
    return _run(
        "enhance_image_color_advance",
        lambda f: m.EnhanceImageColorAdvanceRequest(
            image_urlobject=f, mode=mode, output_format="jpg"),
        "image_url", image_path)


def color_chain(image_path: Path, *, modes: tuple[str, ...] = ("Rec709", "ln17_256")) -> bytes:
    """Apply several color-enhance modes in sequence, each on the previous result.
    Returns the final JPEG bytes; one API call per mode."""
    import tempfile

    src = image_path
    data = b""
    tmps: list[Path] = []
    try:
        for i, mode in enumerate(modes):
            data = color_enhance(src, mode=mode)
            if i < len(modes) - 1:          # feed this result into the next mode
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tf.write(data)
                tf.close()
                src = Path(tf.name)
                tmps.append(src)
        return data
    finally:
        for t in tmps:
            t.unlink(missing_ok=True)


def fit_jpeg(src: Path, dest: Path, *, max_long: int, max_short: int,
             max_bytes: int = MAX_BYTES) -> tuple[int, int]:
    """Orient, downscale to fit (max_long × max_short), re-encode JPEG < max_bytes.
    Saves to dest and returns the final (w, h). Shared by the test harness and the bot."""
    from PIL import Image

    from ic2x.utils.image_utils import oriented

    with Image.open(src) as im:
        im = oriented(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_long / max(w, h), max_short / min(w, h))
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        q = 95
        while True:
            im.save(dest, "JPEG", quality=q)
            if dest.stat().st_size <= max_bytes or q <= 70:
                break
            q -= 8
        return im.size


def enhance_post_image(src: Path, dest: Path, *, mode: str = "ln17_256",
                       max_edge: int = MAX_LONG_EDGE) -> bool:
    """Color-enhance src → dest (JPEG), downscaling to the API input limits first.

    FAIL-OPEN: on ANY failure (creds, not-activated, size, timeout, network) it logs a
    warning, leaves dest untouched, and returns False so the caller posts the original.
    Returns True only when dest now holds the enhanced image. src may equal dest (the
    source is read into a temp before dest is written, so in-place is safe)."""
    import tempfile

    tmp = None
    try:
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tf.close()
        tmp = Path(tf.name)
        fit_jpeg(src, tmp, max_long=max_edge, max_short=round(max_edge * 1080 / 1920))
        data = color_enhance(tmp, mode=mode)
        dest.write_bytes(data)
        return True
    except Exception as exc:  # noqa: BLE001 — enhancement must NEVER block a post
        logger.warning("color enhance failed (%s) — posting original", exc)
        return False
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# Capability registry — name → callable(image_path) -> enhanced JPEG bytes.
# Extensible: add more VIAPI actions or parameter variants to "try different things".
CAPABILITIES = {
    "superres": lambda p: superres(p, upscale=1),
    "color": lambda p: color_enhance(p, mode="Rec709"),
    "color-strong": lambda p: color_enhance(p, mode="ln17_256"),
    "color-both": lambda p: color_chain(p, modes=("Rec709", "ln17_256")),
}
