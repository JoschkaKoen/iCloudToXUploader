from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps


def encode_image_b64(path: Path) -> str:
    """Re-encode any image format as base64 JPEG for the OpenAI vision API.

    Applies exif_transpose so orientation is baked in before encoding.
    Works for both inbox files (HEIC/PNG) and prepared queue files (JPEG).
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()
