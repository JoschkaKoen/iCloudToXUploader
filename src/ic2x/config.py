"""
Load configuration from .env (secrets) and config.yaml (non-secret settings).
All other modules import from here — nothing reads env vars or yaml directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/ic2x/config.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class Config:
    # iCloud
    icloud_username: str
    icloud_password: str
    icloud_recent_count: int
    icloud_cookie_dir: Path

    # Paths
    inbox_dir: Path
    queue_dir: Path
    approved_dir: Path
    posted_dir: Path
    rejected_dir: Path
    db_path: Path
    logs_dir: Path

    # Proxy
    proxy_http: str
    proxy_https: str

    # Gemini
    gemini_api_key: str
    gemini_safety_model: str
    gemini_quality_model: str

    # X / Twitter
    twitter_consumer_key: str
    twitter_consumer_secret: str
    twitter_access_token: str
    twitter_access_token_secret: str
    x_dry_run: bool

    # Limits
    daily_gemini_calls: int
    hamming_threshold: int

    # Enhancement
    enhance_enabled: bool
    enhance_instructir_dir: Path
    enhance_prompt: str

    # Review
    auto_approve: bool


def _require_env(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise ValueError(
            f"Missing required environment variable: {key}\n"
            f"Add it to .env in the project root (see .env.example)."
        )
    return val


def _resolve(p: str) -> Path:
    """Expand ~ and make relative paths absolute from the project root."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def load_config(config_path: Path | None = None) -> Config:
    path = config_path or (_PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    icloud_password = _require_env("ICLOUD_PASSWORD")
    gemini_api_key = _require_env("GEMINI_API_KEY")
    twitter_consumer_key    = _require_env("TWITTER_CONSUMER_KEY")
    twitter_consumer_secret = _require_env("TWITTER_CONSUMER_SECRET")
    twitter_access_token    = _require_env("TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret = _require_env("TWITTER_ACCESS_TOKEN_SECRET")

    return Config(
        # iCloud
        icloud_username=y["icloud"]["username"],
        icloud_password=icloud_password,
        icloud_recent_count=int(y["icloud"]["recent_count"]),
        icloud_cookie_dir=_resolve(y["icloud"]["cookie_dir"]),

        # Paths
        inbox_dir=_resolve(y["paths"]["inbox"]),
        queue_dir=_resolve(y["paths"]["queue"]),
        approved_dir=_resolve(y["paths"]["approved"]),
        posted_dir=_resolve(y["paths"]["posted"]),
        rejected_dir=_resolve(y["paths"]["rejected"]),
        db_path=_resolve(y["paths"]["db"]),
        logs_dir=_resolve(y["paths"]["logs"]),

        # Proxy
        proxy_http=y.get("proxy", {}).get("http", ""),
        proxy_https=y.get("proxy", {}).get("https", ""),

        # Gemini
        gemini_api_key=gemini_api_key,
        gemini_safety_model=y["gemini"]["safety_model"],
        gemini_quality_model=y["gemini"]["quality_model"],

        # X
        twitter_consumer_key=twitter_consumer_key,
        twitter_consumer_secret=twitter_consumer_secret,
        twitter_access_token=twitter_access_token,
        twitter_access_token_secret=twitter_access_token_secret,
        x_dry_run=bool(y["x"].get("dry_run", True)),

        # Limits
        daily_gemini_calls=int(y["limits"]["daily_gemini_calls"]),
        hamming_threshold=int(y["limits"]["hamming_threshold"]),

        # Enhancement
        enhance_enabled=bool(y["enhance"].get("enabled", False)),
        enhance_instructir_dir=_resolve(y["enhance"]["instructir_dir"]),
        enhance_prompt=y["enhance"].get("prompt", ""),

        # Review
        auto_approve=bool(y["review"].get("auto_approve", False)),
    )


def ensure_dirs(cfg: Config) -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        cfg.inbox_dir,
        cfg.queue_dir,
        cfg.approved_dir,
        cfg.posted_dir,
        cfg.rejected_dir / "duplicate",
        cfg.rejected_dir / "screenshot",
        cfg.rejected_dir / "safety",
        cfg.rejected_dir / "quality",
        cfg.rejected_dir / "gemini_refused",
        cfg.logs_dir,
        cfg.icloud_cookie_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
