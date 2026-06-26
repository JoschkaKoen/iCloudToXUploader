"""
Winner caption pass — one VLM call that captions the chosen photo, grounded in WHERE and
WHEN it was taken.

The burst judge only sees EXIF-stripped thumbnails, so it can't know the city. This pass
runs after the winner's original is downloaded — once the reverse-geocoded city and local
time are known — and is told both in a single prompt. Best-effort: any failure returns
(None, used) and the caller keeps the judge's caption. The 📍 line is added by the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ic2x.config import Config
from ic2x.utils.ai_client import (
    JudgeCall,
    call_vision_judge,
    parse_model_effort,
    provider_for_model,
)

logger = logging.getLogger("ic2x.caption")

# One prompt; the photo's place and time are interpolated in. {{ }} are literal JSON braces.
_CAPTION_PROMPT = """You are an expat who has lived in China for many years, captioning your own photo on twitter (X)
for people from Western countries on social media platforms that have never visited China 
and don't have a very accurate image of China. The photo you are seeing was taken in {place} at {when}.

Create a sentence for a tweet for this image. The image will be posted along with your sentence on twitter (X).
If possible share an insight about China about what you can see in the image. Something 
the viewer can learn from the image. 
Make the caption interesting. Don't make it overly positive to stay authentic. The viewer should judge himself. 
But don't make the tweet negative, too. 
The tweet needs to be compatible with Chinese culture and laws! The tweet needs to be China compliant 
(I am living in China).

Say only what you can see or genuinely know — don't invent prices. 

Under 200 characters, no hashtags, ending with one fitting emoji.

Return ONLY JSON: {{"caption": "<caption>"}}"""


def generate_caption(image_path: Path, place: str | None, when: str | None,
                     cfg: Config) -> tuple[str | None, bool]:
    """Caption the winner image, grounded in `place` + `when`. Returns (caption, used_net);
    caption is None on any failure (caller keeps the judge's caption). Uses the same cloud
    vision model as the judge; skipped on a local-only/ollama model."""
    model, _ = parse_model_effort(cfg.judge_model)
    if provider_for_model(model) == "ollama":
        return None, False  # the caption pass needs a cloud VLM

    prompt = _CAPTION_PROMPT.format(place=place or "an unknown place",
                                    when=when or "an unknown time")
    parsed, _elapsed, ok, used = call_vision_judge(
        model_string=cfg.judge_model,
        ollama_base_url=cfg.ollama_base_url,
        call=JudgeCall(
            image_path=image_path,
            prompt=prompt,
            max_px=cfg.judge_image_max_px,
            fail_value={"caption": ""},
            refused_value={"caption": ""},
            label="caption",
        ),
    )
    if not ok:
        return None, used

    caption = " ".join((parsed.get("caption") or "").split())  # normalise whitespace
    return (caption or None), used
