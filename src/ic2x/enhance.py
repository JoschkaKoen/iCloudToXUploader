"""
Optional InstructIR post-enhancement for queued images.

Ported from XBot-3/services/instructir_enhance.py.
Runs InstructIR in a subprocess so the CUDA context is fully released
when the subprocess exits, leaving VRAM free for other operations.

Requires a local clone of https://github.com/mv-lab/InstructIR and weights
(im_instructir-7d.pt, lm_instructir-7d.pt in the repo root, or
auto-downloaded from HuggingFace when missing).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("ic2x.enhance")

_MODEL_NAME = "im_instructir-7d.pt"
_LM_MODEL   = "lm_instructir-7d.pt"
_CONFIG_REL = Path("configs") / "eval5d.yml"

# Subprocess script — runs inside a child process for clean CUDA context isolation.
# Called as: python -c _ENHANCE_CODE <image_path> <ir_dir> <prompt> <output_path>
_ENHANCE_CODE = r"""
import sys, os, argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
import tempfile

def _dict2namespace(config):
    ns = argparse.Namespace()
    for key, value in config.items():
        setattr(ns, key, _dict2namespace(value) if isinstance(value, dict) else value)
    return ns

def _torch_load(path):
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)

image_path  = sys.argv[1]
ir_dir      = Path(sys.argv[2])
prompt      = sys.argv[3]
output_path = sys.argv[4]

root_str = str(ir_dir)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from models import instructir
from text.models import LanguageModel, LMHead

cfg_path = ir_dir / "configs" / "eval5d.yml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = _dict2namespace(yaml.safe_load(f))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = instructir.create_model(
    input_channels=cfg.model.in_ch,
    width=cfg.model.width,
    enc_blks=cfg.model.enc_blks,
    middle_blk_num=cfg.model.middle_blk_num,
    dec_blks=cfg.model.dec_blks,
    txtdim=cfg.model.textdim,
).to(device)
model.load_state_dict(_torch_load(ir_dir / "im_instructir-7d.pt"), strict=True)
model.eval()

language_model = LanguageModel(model=cfg.llm.model)
lm_head = LMHead(
    embedding_dim=cfg.llm.model_dim,
    hidden_dim=cfg.llm.embd_dim,
    num_classes=cfg.llm.nclasses,
).to(device)
lm_head.load_state_dict(_torch_load(ir_dir / "lm_instructir-7d.pt"), strict=True)
lm_head.eval()

with Image.open(image_path) as im:
    im = im.convert("RGB")
    original_size = im.size
    pil_in = im.copy()

img = np.array(pil_in).astype(np.float32) / 255.0
y   = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

lm_embd = language_model(prompt).to(device)
with torch.no_grad():
    text_embd, _ = lm_head(lm_embd)
    x_hat = model(y, text_embd)

restored = x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
restored = (np.clip(restored, 0.0, 1.0) * 255.0).round().astype(np.uint8)
result   = Image.fromarray(restored)

if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)

out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=out_dir)
os.close(fd)
result.save(tmp_path, format="JPEG", quality=92)
os.replace(tmp_path, output_path)
print(f"enhanced:{output_path}", flush=True)
"""


def _ensure_weights(ir_dir: Path) -> None:
    im_path = ir_dir / _MODEL_NAME
    lm_path = ir_dir / _LM_MODEL
    if im_path.is_file() and lm_path.is_file():
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Missing {_MODEL_NAME} / {_LM_MODEL} under {ir_dir} and "
            "huggingface_hub is not installed. Install huggingface-hub or "
            "copy checkpoints from https://huggingface.co/marcosv/InstructIR"
        ) from exc
    logger.info("enhance: downloading InstructIR checkpoints into %s …", ir_dir)
    hf_hub_download(repo_id="marcosv/InstructIR", filename=_MODEL_NAME, local_dir=str(ir_dir))
    hf_hub_download(repo_id="marcosv/InstructIR", filename=_LM_MODEL,   local_dir=str(ir_dir))


def enhance_image(
    path: Path,
    ir_dir: Path,
    prompt: str,
    enabled: bool = True,
) -> Path:
    """
    Run InstructIR on path in a subprocess.
    Returns the enhanced file path on success, or the original path on
    skip/failure (never raises — the pipeline must never fail because of this).
    """
    if not enabled:
        return path

    ir_dir = ir_dir.expanduser().resolve()

    if not ir_dir.is_dir():
        logger.warning("enhance: INSTRUCTIR_DIR not found: %s — skipping", ir_dir)
        return path

    if not (ir_dir / _CONFIG_REL).is_file():
        logger.warning("enhance: missing config %s — skipping", ir_dir / _CONFIG_REL)
        return path

    if not prompt.strip():
        logger.warning("enhance: empty prompt — skipping")
        return path

    try:
        _ensure_weights(ir_dir)
    except Exception as exc:
        logger.warning("enhance: could not obtain weights (%s) — skipping", exc)
        return path

    out_path = path.parent / (path.stem + "_enhanced" + path.suffix)

    logger.info("enhance: enhancing %s (subprocess) …", path.name)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _ENHANCE_CODE,
             str(path), str(ir_dir), prompt, str(out_path)],
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("enhance: subprocess launch failed (%s) — skipping", exc)
        return path

    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-600:]
        logger.warning("enhance: subprocess exited %d:\n%s", result.returncode, tail)
        return path

    logger.info("enhance: %s → %s — VRAM fully released", path.name, out_path.name)
    return out_path
