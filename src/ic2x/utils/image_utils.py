from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


_HEIF_OPS = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_270,
    7: Image.TRANSVERSE,
    8: Image.ROTATE_90,
}


def oriented(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation (JPEG) or HEIF orientation (HEIC) to img.

    Used by every site that decodes a source image: dedup (pHash), prepare
    (re-encoded JPEG), and encode_image_b64 (AI judges). Keeping a single
    implementation guarantees the same pixels reach every consumer.
    """
    img = ImageOps.exif_transpose(img)          # standard JPEG EXIF path
    orientation = img.info.get("orientation")   # pillow_heif fallback (int 1-8)
    if orientation and orientation != 1:
        op = _HEIF_OPS.get(orientation)
        if op:
            img = img.transpose(op)
    return img


def encode_image_b64(path: Path, *, max_px: int | None = None) -> str:
    """Re-encode any image format as base64 JPEG for AI vision APIs.

    Applies the shared `oriented()` helper so orientation is baked in before
    encoding. Works for both inbox files (HEIC/PNG) and prepared queue files
    (JPEG).

    If *max_px* is set, the image is downscaled so its long edge ≤ max_px
    (aspect ratio preserved). Upscaling never occurs.
    """
    with Image.open(path) as img:
        img = oriented(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if max_px is not None:
            w, h = img.size
            long_edge = max(w, h)
            if long_edge > max_px:
                scale = max_px / long_edge
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()
