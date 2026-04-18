"""
Load configuration from default.env (committed defaults) + .env (secrets).
All other modules import from here — nothing reads env vars directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_project_env() -> None:
    """Load default.env (committed defaults) then .env (secrets, override=True)."""
    default_env = _PROJECT_ROOT / "default.env"
    if default_env.exists():
        load_dotenv(default_env, override=False)
    load_dotenv(_PROJECT_ROOT / ".env", override=True)


load_project_env()


@dataclass
class Config:
    # iCloud
    icloud_username: str
    icloud_password: str
    icloud_recent_count: int
    icloud_until_found: int   # passes --until-found to icloudpd; 0 = disabled
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

    # X / Twitter
    twitter_consumer_key: str
    twitter_consumer_secret: str
    twitter_access_token: str
    twitter_access_token_secret: str
    x_dry_run: bool

    # Limits
    daily_ai_calls: int
    hamming_threshold: int

    # Enhancement
    enhance_enabled: bool
    enhance_instructir_dir: Path
    enhance_prompt: str

    # Review
    auto_approve: bool

    # Daemon
    post_interval_hours: int
    daemon_check_interval: int
    icloud_lookback_max: int


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


def _bool_env(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")


def load_config() -> Config:
    return Config(
        # iCloud
        icloud_username=_require_env("ICLOUD_USERNAME"),
        icloud_password=_require_env("ICLOUD_PASSWORD"),
        icloud_recent_count=int(os.getenv("ICLOUD_RECENT_COUNT", "20")),
        icloud_until_found=int(os.getenv("ICLOUD_UNTIL_FOUND", "3")),
        icloud_cookie_dir=_resolve(os.getenv("ICLOUD_COOKIE_DIR", "./icloud_auth")),

        # Paths
        inbox_dir=_resolve(os.getenv("INBOX_DIR", "./inbox")),
        queue_dir=_resolve(os.getenv("QUEUE_DIR", "./queue")),
        approved_dir=_resolve(os.getenv("APPROVED_DIR", "./approved")),
        posted_dir=_resolve(os.getenv("POSTED_DIR", "./posted")),
        rejected_dir=_resolve(os.getenv("REJECTED_DIR", "./rejected")),
        db_path=_resolve(os.getenv("DB_PATH", "./state.db")),
        logs_dir=_resolve(os.getenv("LOGS_DIR", "./logs")),

        # Proxy
        proxy_http=os.getenv("IC2X_HTTP_PROXY", ""),
        proxy_https=os.getenv("IC2X_HTTPS_PROXY", ""),

        # X
        twitter_consumer_key=_require_env("TWITTER_CONSUMER_KEY"),
        twitter_consumer_secret=_require_env("TWITTER_CONSUMER_SECRET"),
        twitter_access_token=_require_env("TWITTER_ACCESS_TOKEN"),
        twitter_access_token_secret=_require_env("TWITTER_ACCESS_TOKEN_SECRET"),
        x_dry_run=_bool_env("X_DRY_RUN", default=True),

        # Limits
        daily_ai_calls=int(os.getenv("DAILY_AI_CALLS", "200")),
        hamming_threshold=int(os.getenv("HAMMING_THRESHOLD", "12")),

        # Enhancement
        enhance_enabled=_bool_env("ENHANCE_ENABLED", default=False),
        enhance_instructir_dir=_resolve(os.getenv("ENHANCE_INSTRUCTIR_DIR", "~/Programming/InstructIR")),
        enhance_prompt=os.getenv("ENHANCE_PROMPT", "sharpen this image, remove blur, improve clarity"),

        # Review
        auto_approve=_bool_env("AUTO_APPROVE", default=False),

        # Daemon
        post_interval_hours=int(os.getenv("POST_INTERVAL_HOURS", "8")),
        daemon_check_interval=int(os.getenv("DAEMON_CHECK_INTERVAL", "1800")),
        icloud_lookback_max=int(os.getenv("ICLOUD_LOOKBACK_MAX", "200")),
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
        cfg.rejected_dir / "model_refused",
        cfg.logs_dir,
        cfg.icloud_cookie_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
