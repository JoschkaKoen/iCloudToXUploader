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
    twitter_bearer_token: str        # app-only auth — the only one that reads tweets (get_tweets)
    x_dry_run: bool
    test_mode: bool                 # IC2X_TEST_MODE / --test: dry-run + fast loop + isolated test_run/ state

    # Location stamp (GPS EXIF → "📍 City, Country" caption line)
    location_enabled: bool
    location_timeout: float
    caption_pass_enabled: bool      # location/time-grounded caption pass on the winner
    caption_candidates: int         # best-of-N captions (N parallel calls + AI picker)
    caption_picker_model: str       # CAPTION_PICKER_MODEL or judge_model

    # Limits / dedup
    daily_ai_calls: int             # also bounds walk-back cost per day
    hamming_threshold: int          # cross-library pHash dedup distance
    quality_min_score: int          # posting bar: judge's 0-10 quality score floor

    # Burst assembly
    burst_hamming_threshold: int    # consecutive-shot similarity (tighter than dedup)
    burst_max_size: int             # images per burst sent to the judge (token guard)
    burst_max_attempts: int         # pre-commit failures before SEEN(error) (poison-burst breaker)
    thumb_version: str              # pyicloud rendition for burst eval: thumb | medium
    screenshot_album_refresh_cycles: int  # cycles between Screenshots-album flag refreshes
    prefetch_concurrency: int       # thumbnails downloaded+hashed in parallel per batch (1 = serial)

    # Posting / loop
    post_interval_hours: float      # fractional hours allowed (e.g. 23000s = 6.388...h)
    daemon_check_interval: int      # loop poll granularity (seconds)
    max_posts_per_day: int          # rolling-24h safety backstop
    post_max_attempts: int          # flush retries before REJECTED(post_failed)
    reauth_notify_cmd: str          # shell cmd run on ReauthRequired

    # Enhancement (opt-in)
    enhance_enabled: bool
    enhance_instructir_dir: Path
    enhance_prompt: str

    # Local adaptive polish (free, CPU-only PIL/numpy — opt-in; composes before the
    # paid Aliyun color enhance, or stands alone as a zero-cost alternative)
    polish_enabled: bool
    polish_intensity: str           # natural | punchy | off

    # Rotation (opt-in — prepare() already bakes EXIF orientation)
    rotation_enabled: bool
    rotation_model: str             # ROTATION_MODEL or default_model
    rotation_image_max_px: int | None

    # Scene dedup + grouping (cheap VLM "same scene/object" — dedup vs recent posts,
    # and grouping at burst-assembly to merge orientation/angle variants into 1 scene)
    scene_dedup_enabled: bool
    scene_group_enabled: bool
    scene_dedup_model: str
    scene_dedup_recent_n: int
    scene_dedup_image_max_px: int | None
    scene_thumbs_dir: Path

    # Local (Ollama) safety pre-filter — screens bursts ON-DEVICE before any
    # thumbnail is uploaded to a cloud judge; fail-through on any problem
    local_prefilter_enabled: bool
    local_prefilter_model: str
    local_prefilter_min_free_gb: int    # skip local inference below this kernel-free level
    local_judge_shadow: bool            # Tier-2 pilot: log local-vs-cloud judge agreement
    local_judge_model: str

    # Owner-selfie gate (reference-based "is the owner the main subject?" check on
    # the winner; the judge's composition rule (flag "selfie") is the first layer)
    owner_check_enabled: bool
    owner_check_model: str           # OWNER_CHECK_MODEL or rotation_model
    owner_check_image_max_px: int | None
    owner_refs_dir: Path             # reference photos of the owner's face

    # Color enhancement (Aliyun VIAPI EnhanceImageColor on the winner before posting)
    color_enhance_enabled: bool
    color_enhance_mode: str          # Rec709 (natural) | ln17_256 (stronger) | LogC
    color_enhance_max_edge: int      # downscale long edge before sending (API input cap)
    color_enhance_cost_rmb: float    # per call beyond the free monthly quota (cost tracking)
    color_enhance_free_quota: int    # free calls per calendar month (Aliyun gives 100)

    # Startup reconciliation (sync DB with X; re-queue posts deleted on X)
    reconcile_on_startup: bool
    reconcile_recent_n: int          # how many recent X posts to fetch
    reconcile_max_requeue_per_run: int  # sanity cap — skip if more look deleted

    # AI models / vision
    default_model: str             # AI_DEFAULT_MODEL (with optional ", effort" suffix)
    judge_model: str               # JUDGE_MODEL or default_model
    judge_extra_rules: str         # owner-specific do-not-post rules appended to the judge prompt
    ollama_base_url: str
    judge_image_max_px: int | None       # cloud providers; None = full-res (rare)
    ollama_image_max_px: int | None      # all Ollama calls (memory-bound)


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


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {key}={raw!r} is not a valid number"
        )


def load_config() -> Config:
    default_model = os.getenv("AI_DEFAULT_MODEL", "").strip() or _DEFAULT_AI_MODEL
    judge_model    = os.getenv("JUDGE_MODEL",    "").strip() or default_model
    rotation_model = os.getenv("ROTATION_MODEL", "").strip() or default_model

    return Config(
        # iCloud — soft here; validated by require_credentials() only for commands
        # that actually hit iCloud, so offline diagnostics + `ic2x cost` run without them.
        icloud_username=os.getenv("ICLOUD_USERNAME", "").strip(),
        icloud_password=os.getenv("ICLOUD_PASSWORD", "").strip(),
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

        # X — soft here; only REAL posting needs these (dry-run skips them), so
        # require_credentials() validates them just-in-time.
        twitter_consumer_key=os.getenv("TWITTER_CONSUMER_KEY", "").strip(),
        twitter_consumer_secret=os.getenv("TWITTER_CONSUMER_SECRET", "").strip(),
        twitter_access_token=os.getenv("TWITTER_ACCESS_TOKEN", "").strip(),
        twitter_access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "").strip(),
        twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN", "").strip(),
        x_dry_run=_bool_env("X_DRY_RUN", default=True),
        test_mode=_bool_env("IC2X_TEST_MODE", default=False),

        # Location stamp (GPS EXIF → "📍 City, Country")
        location_enabled=_bool_env("LOCATION_ENABLED", default=True),
        location_timeout=float(_int_env("LOCATION_TIMEOUT", 10)),
        caption_pass_enabled=_bool_env("CAPTION_PASS_ENABLED", default=True),
        caption_candidates=_int_env("CAPTION_CANDIDATES", 4),
        caption_picker_model=os.getenv("CAPTION_PICKER_MODEL", "").strip() or judge_model,

        # Limits / dedup
        daily_ai_calls=_int_env("DAILY_AI_CALLS", 200),
        hamming_threshold=_int_env("HAMMING_THRESHOLD", 12),
        quality_min_score=_int_env("QUALITY_MIN_SCORE", 7),

        # Burst assembly
        burst_hamming_threshold=_int_env("BURST_HAMMING_THRESHOLD", 8),
        burst_max_size=_int_env("BURST_MAX_SIZE", 5),
        burst_max_attempts=_int_env("BURST_MAX_ATTEMPTS", 3),
        thumb_version=os.getenv("THUMB_VERSION", "").strip() or "thumb",
        screenshot_album_refresh_cycles=_int_env("SCREENSHOT_ALBUM_REFRESH", 20),
        prefetch_concurrency=_int_env("PREFETCH_CONCURRENCY", 6),

        # Posting / loop
        post_interval_hours=_float_env("POST_INTERVAL_HOURS", 5.0),
        daemon_check_interval=_int_env("DAEMON_CHECK_INTERVAL", 1800),
        max_posts_per_day=_int_env("MAX_POSTS_PER_DAY", 6),
        post_max_attempts=_int_env("POST_MAX_ATTEMPTS", 3),
        reauth_notify_cmd=os.getenv("REAUTH_NOTIFY_CMD", ""),

        # Enhancement
        enhance_enabled=_bool_env("ENHANCE_ENABLED", default=False),
        enhance_instructir_dir=_resolve(os.getenv("ENHANCE_INSTRUCTIR_DIR", "~/Programming/InstructIR")),
        enhance_prompt=os.getenv("ENHANCE_PROMPT", "sharpen this image, remove blur, improve clarity"),

        # Local adaptive polish
        polish_enabled=_bool_env("POLISH_ENABLED", default=False),
        polish_intensity=os.getenv("POLISH_INTENSITY", "").strip() or "natural",

        # Rotation
        rotation_enabled=_bool_env("ROTATION_ENABLED", default=False),
        rotation_model=rotation_model,
        rotation_image_max_px=_opt_int_env("ROTATION_IMAGE_MAX_PX", _DEFAULT_JUDGE_MAX_PX),

        # Scene dedup + grouping
        scene_dedup_enabled=_bool_env("SCENE_DEDUP_ENABLED", default=False),
        scene_group_enabled=_bool_env("SCENE_GROUP_ENABLED", default=False),
        scene_dedup_model=os.getenv("SCENE_DEDUP_MODEL", "").strip() or "qwen3-vl-flash",
        scene_dedup_recent_n=_int_env("SCENE_DEDUP_RECENT_N", 6),
        scene_dedup_image_max_px=_opt_int_env("SCENE_DEDUP_IMAGE_MAX_PX", 384),
        scene_thumbs_dir=_resolve(os.getenv("SCENE_THUMBS_DIR", "./scene_thumbs")),
        local_prefilter_enabled=_bool_env("LOCAL_PREFILTER_ENABLED", default=False),
        local_prefilter_model=os.getenv("LOCAL_PREFILTER_MODEL", "").strip() or "qwen3-vl:8b",
        local_prefilter_min_free_gb=_int_env("LOCAL_PREFILTER_MIN_FREE_GB", 5),
        local_judge_shadow=_bool_env("LOCAL_JUDGE_SHADOW", default=False),
        local_judge_model=os.getenv("LOCAL_JUDGE_MODEL", "").strip() or "qwen3-vl:8b",
        owner_check_enabled=_bool_env("OWNER_CHECK_ENABLED", default=True),
        owner_check_model=os.getenv("OWNER_CHECK_MODEL", "").strip() or rotation_model,
        owner_check_image_max_px=_opt_int_env("OWNER_CHECK_IMAGE_MAX_PX", 768),
        owner_refs_dir=_resolve(os.getenv("OWNER_REFS_DIR", "./owner_refs")),

        # Color enhancement (EnhanceImageColor on the winner before posting)
        color_enhance_enabled=_bool_env("COLOR_ENHANCE_ENABLED", default=False),
        color_enhance_mode=os.getenv("COLOR_ENHANCE_MODE", "").strip() or "ln17_256",
        color_enhance_max_edge=_int_env("COLOR_ENHANCE_MAX_EDGE", 1920),
        color_enhance_cost_rmb=float(os.getenv("COLOR_ENHANCE_COST_RMB", "").strip() or "0.02"),
        color_enhance_free_quota=_int_env("COLOR_ENHANCE_FREE_QUOTA", 100),

        # Startup reconciliation
        reconcile_on_startup=_bool_env("RECONCILE_ON_STARTUP", default=True),
        reconcile_recent_n=_int_env("RECONCILE_RECENT_N", 50),
        reconcile_max_requeue_per_run=_int_env("RECONCILE_MAX_REQUEUE_PER_RUN", 20),

        # AI models / vision
        default_model=default_model,
        judge_model=judge_model,
        judge_extra_rules=os.getenv("JUDGE_EXTRA_RULES", "").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "").strip() or _DEFAULT_OLLAMA_URL,
        judge_image_max_px=_opt_int_env("JUDGE_IMAGE_MAX_PX", _DEFAULT_JUDGE_MAX_PX),
        ollama_image_max_px=_opt_int_env("OLLAMA_IMAGE_MAX_PX", None),
    )


def require_credentials(cfg: Config, *, icloud: bool = True, x: bool = True) -> None:
    """Raise ValueError (clear, aggregated) if credentials needed for THIS run are
    missing. iCloud creds are needed to fetch photos; X creds only for REAL posts
    (pass x=False in dry-run). Offline commands call neither check, so they run with
    no creds at all."""
    missing: list[str] = []
    if icloud:
        if not cfg.icloud_username:
            missing.append("ICLOUD_USERNAME")
        if not cfg.icloud_password:
            missing.append("ICLOUD_PASSWORD")
    if x:
        for name, val in (
            ("TWITTER_CONSUMER_KEY", cfg.twitter_consumer_key),
            ("TWITTER_CONSUMER_SECRET", cfg.twitter_consumer_secret),
            ("TWITTER_ACCESS_TOKEN", cfg.twitter_access_token),
            ("TWITTER_ACCESS_TOKEN_SECRET", cfg.twitter_access_token_secret),
        ):
            if not val:
                missing.append(name)
    if missing:
        raise ValueError(
            "Missing required credential(s) in .env: " + ", ".join(missing)
            + "\nAdd them to .env in the project root (see .env.example)."
        )


def ensure_dirs(cfg: Config) -> None:
    """Create only the always-needed dirs: logs + the iCloud cookie/session dir. The
    content dirs (work, queue, approved, posted, scene_thumbs, reviewed) are created
    lazily by their writers, so a run never leaves empty folders behind."""
    for d in (cfg.logs_dir, cfg.icloud_cookie_dir):
        d.mkdir(parents=True, exist_ok=True)
