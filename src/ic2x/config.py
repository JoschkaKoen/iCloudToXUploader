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

_DEFAULT_AI_MODEL    = "gemini-2.5-flash-lite"
_DEFAULT_OLLAMA_URL  = "http://localhost:11434/v1"
_DEFAULT_JUDGE_MAX_PX = 1024


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
    icloud_cookie_dir: Path
    icloud_with_family: bool        # include the shared family library (unsafe for auto-post)
    icloud_family_override: bool    # I_KNOW_FAMILY_PHOTOS_ARE_PUBLIC — required to allow with_family
    icloud_china_mainland: bool     # route to the China-mainland Apple endpoint

    # Paths
    work_dir: Path                  # scratch for thumbnail + original downloads
    queue_dir: Path
    approved_dir: Path
    posted_dir: Path
    db_path: Path
    logs_dir: Path

    # Observability
    keep_reviewed: bool             # save each judged thumbnail to reviewed_dir
    reviewed_dir: Path              # browseable feed of decisions (named by outcome)

    # Proxy
    proxy_http: str
    proxy_https: str

    # X / Twitter
    twitter_consumer_key: str
    twitter_consumer_secret: str
    twitter_access_token: str
    twitter_access_token_secret: str
    x_dry_run: bool

    # Limits / dedup
    daily_ai_calls: int             # also bounds walk-back cost per day
    hamming_threshold: int          # cross-library pHash dedup distance

    # Burst assembly
    burst_hamming_threshold: int    # consecutive-shot similarity (tighter than dedup)
    burst_max_size: int             # images per burst sent to the judge (token guard)
    burst_max_attempts: int         # pre-commit failures before SEEN(error) (poison-burst breaker)
    thumb_version: str              # pyicloud rendition for burst eval: thumb | medium
    screenshot_album_refresh_cycles: int  # cycles between Screenshots-album flag refreshes

    # Posting / loop
    post_interval_hours: int
    daemon_check_interval: int      # loop poll granularity (seconds)
    max_posts_per_day: int          # rolling-24h safety backstop
    post_max_attempts: int          # flush retries before REJECTED(post_failed)
    reauth_notify_cmd: str          # shell cmd run on ReauthRequired

    # Enhancement (opt-in)
    enhance_enabled: bool
    enhance_instructir_dir: Path
    enhance_prompt: str

    # Rotation (opt-in — prepare() already bakes EXIF orientation)
    rotation_enabled: bool
    rotation_model: str             # ROTATION_MODEL or default_model
    rotation_image_max_px: int | None

    # AI models / vision
    default_model: str             # AI_DEFAULT_MODEL (with optional ", effort" suffix)
    judge_model: str               # JUDGE_MODEL or default_model
    ollama_base_url: str
    judge_image_max_px: int | None       # cloud providers; None = full-res (rare)
    ollama_image_max_px: int | None      # all Ollama calls (memory-bound)


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


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {key}={raw!r} is not a valid integer"
        )


def _opt_int_env(key: str, default: int | None) -> int | None:
    """Like _int_env but returns None when the env var is unset and default is None."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {key}={raw!r} is not a valid integer"
        )


def load_config() -> Config:
    default_model = os.getenv("AI_DEFAULT_MODEL", "").strip() or _DEFAULT_AI_MODEL
    judge_model    = os.getenv("JUDGE_MODEL",    "").strip() or default_model
    rotation_model = os.getenv("ROTATION_MODEL", "").strip() or default_model

    return Config(
        # iCloud
        icloud_username=_require_env("ICLOUD_USERNAME"),
        icloud_password=_require_env("ICLOUD_PASSWORD"),
        icloud_cookie_dir=_resolve(os.getenv("ICLOUD_COOKIE_DIR", "./icloud_auth")),
        icloud_with_family=_bool_env("ICLOUD_WITH_FAMILY", default=False),
        icloud_family_override=_bool_env("I_KNOW_FAMILY_PHOTOS_ARE_PUBLIC", default=False),
        icloud_china_mainland=_bool_env("ICLOUD_CHINA_MAINLAND", default=False),

        # Paths
        work_dir=_resolve(os.getenv("WORK_DIR", "./work")),
        queue_dir=_resolve(os.getenv("QUEUE_DIR", "./queue")),
        approved_dir=_resolve(os.getenv("APPROVED_DIR", "./approved")),
        posted_dir=_resolve(os.getenv("POSTED_DIR", "./posted")),
        db_path=_resolve(os.getenv("DB_PATH", "./state.db")),
        logs_dir=_resolve(os.getenv("LOGS_DIR", "./logs")),

        # Observability
        keep_reviewed=_bool_env("KEEP_REVIEWED", default=True),
        reviewed_dir=_resolve(os.getenv("REVIEWED_DIR", "./reviewed")),

        # Proxy
        proxy_http=os.getenv("IC2X_HTTP_PROXY", ""),
        proxy_https=os.getenv("IC2X_HTTPS_PROXY", ""),

        # X
        twitter_consumer_key=_require_env("TWITTER_CONSUMER_KEY"),
        twitter_consumer_secret=_require_env("TWITTER_CONSUMER_SECRET"),
        twitter_access_token=_require_env("TWITTER_ACCESS_TOKEN"),
        twitter_access_token_secret=_require_env("TWITTER_ACCESS_TOKEN_SECRET"),
        x_dry_run=_bool_env("X_DRY_RUN", default=True),

        # Limits / dedup
        daily_ai_calls=_int_env("DAILY_AI_CALLS", 200),
        hamming_threshold=_int_env("HAMMING_THRESHOLD", 12),

        # Burst assembly
        burst_hamming_threshold=_int_env("BURST_HAMMING_THRESHOLD", 8),
        burst_max_size=_int_env("BURST_MAX_SIZE", 5),
        burst_max_attempts=_int_env("BURST_MAX_ATTEMPTS", 3),
        thumb_version=os.getenv("THUMB_VERSION", "").strip() or "thumb",
        screenshot_album_refresh_cycles=_int_env("SCREENSHOT_ALBUM_REFRESH", 20),

        # Posting / loop
        post_interval_hours=_int_env("POST_INTERVAL_HOURS", 5),
        daemon_check_interval=_int_env("DAEMON_CHECK_INTERVAL", 1800),
        max_posts_per_day=_int_env("MAX_POSTS_PER_DAY", 6),
        post_max_attempts=_int_env("POST_MAX_ATTEMPTS", 3),
        reauth_notify_cmd=os.getenv("REAUTH_NOTIFY_CMD", ""),

        # Enhancement
        enhance_enabled=_bool_env("ENHANCE_ENABLED", default=False),
        enhance_instructir_dir=_resolve(os.getenv("ENHANCE_INSTRUCTIR_DIR", "~/Programming/InstructIR")),
        enhance_prompt=os.getenv("ENHANCE_PROMPT", "sharpen this image, remove blur, improve clarity"),

        # Rotation
        rotation_enabled=_bool_env("ROTATION_ENABLED", default=False),
        rotation_model=rotation_model,
        rotation_image_max_px=_opt_int_env("ROTATION_IMAGE_MAX_PX", _DEFAULT_JUDGE_MAX_PX),

        # AI models / vision
        default_model=default_model,
        judge_model=judge_model,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "").strip() or _DEFAULT_OLLAMA_URL,
        judge_image_max_px=_opt_int_env("JUDGE_IMAGE_MAX_PX", _DEFAULT_JUDGE_MAX_PX),
        ollama_image_max_px=_opt_int_env("OLLAMA_IMAGE_MAX_PX", None),
    )


def ensure_dirs(cfg: Config) -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        cfg.work_dir,
        cfg.queue_dir,
        cfg.approved_dir,
        cfg.posted_dir,
        cfg.logs_dir,
        cfg.reviewed_dir,
        cfg.icloud_cookie_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
