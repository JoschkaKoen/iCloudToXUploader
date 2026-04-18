from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def encode_image_b64(path: Path, *, max_px: int | None = None) -> str:
    """Re-encode any image format as base64 JPEG for AI vision APIs.

    Applies exif_transpose so orientation is baked in before encoding.
    Works for both inbox files (HEIC/PNG) and prepared queue files (JPEG).

    If *max_px* is set, the image is downscaled so its long edge ≤ max_px
    (aspect ratio preserved). Upscaling never occurs.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
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
