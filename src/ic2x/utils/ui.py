"""
Console UI helpers — pretty terminal output for ic2x.
All functions use print() directly so they never appear in log files.
Ported and adapted from XBot-3/utils/ui.py.
"""

import sys
import shutil

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_GRAY   = "\033[90m"
_WHITE  = "\033[97m"

_TOTAL_STAGES = 6  # SHA-256 dedup, screenshot, pHash dedup, safety, quality, prepare


def _w() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 80)


# ── Startup ───────────────────────────────────────────────────────────────────

def startup_banner(cfg) -> None:
    """Printed once at the start of ic2x run."""
    w = _w()
    dry = "YES  ⚠" if cfg.x_dry_run else "no"
    enhance = "enabled" if cfg.enhance_enabled else "disabled"
    print()
    print(f"{_CYAN}{_BOLD}{'═' * w}{_R}")
    print(f"{_CYAN}{_BOLD}  📸  IC2X — iCloud → X Photo Pipeline{_R}")
    print(f"{_CYAN}{'─' * w}{_R}")
    rows = [
        ("👤", "iCloud user",    cfg.icloud_username),
        ("🔢", "Recent count",   str(cfg.icloud_recent_count)),
        ("🤖", "Safety model",   cfg.gemini_safety_model),
        ("🤖", "Quality model",  cfg.gemini_quality_model),
        ("✨", "Enhance",        enhance),
        ("🐦", "Dry run",        dry),
    ]
    for icon, label, value in rows:
        print(f"{_CYAN}  {icon}  {_BOLD}{label:<16}{_R}{_CYAN}{value}{_R}")
    print(f"{_CYAN}{_BOLD}{'═' * w}{_R}")
    print()


# ── Section headers ───────────────────────────────────────────────────────────

def run_banner() -> None:
    w = _w()
    print()
    print(f"{_BOLD}{'━' * w}{_R}")
    print(f"{_BOLD}  📥  PULLING FROM iCLOUD{_R}")
    print(f"{_BOLD}{'━' * w}{_R}")
    print()


def review_section_banner(total: int) -> None:
    w = _w()
    print()
    print(f"{_BOLD}{'━' * w}{_R}")
    print(f"{_BOLD}  🖼   REVIEW QUEUE  ({total} image{'s' if total != 1 else ''}){_R}")
    print(f"{_BOLD}{'━' * w}{_R}")
    print()


def post_section_banner(total: int, dry_run: bool) -> None:
    w = _w()
    tag = "  [DRY RUN]" if dry_run else ""
    print()
    print(f"{_BOLD}{'━' * w}{_R}")
    print(f"{_BOLD}  📤  POSTING{tag}  ({total} image{'s' if total != 1 else ''}){_R}")
    print(f"{_BOLD}{'━' * w}{_R}")
    print()


# ── Per-image headers ─────────────────────────────────────────────────────────

def file_header(filename: str, index: int, total: int) -> None:
    print()
    print(f"  {_BOLD}📷  {filename}{_R}  {_GRAY}({index} / {total}){_R}")


def stage_banner(step: int, name: str) -> None:
    w = _w()
    print(f"  {_CYAN}{'─' * (w - 2)}{_R}")
    print(f"  {_CYAN}▶  [{step}/{_TOTAL_STAGES}]  {name.upper()}{_R}")


# ── Inline status ─────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  {_GREEN}✅  {msg}{_R}")


def info(msg: str) -> None:
    print(f"  {_GRAY}ℹ   {msg}{_R}")


def warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠   {msg}{_R}")


def err(msg: str) -> None:
    print(f"  {_RED}✖   {msg}{_R}", file=sys.stderr)


# ── Decision outputs ──────────────────────────────────────────────────────────

def rejected(filename: str, stage: str, reason) -> None:
    if isinstance(reason, list):
        reason_str = ", ".join(str(r) for r in reason)
    else:
        reason_str = str(reason) if reason else stage
    print(f"  {_RED}✖  {filename} → rejected [{stage}: {reason_str}]{_R}")


def queued(filename: str, phash: str, caption: str) -> None:
    w = _w()
    print(f"  {_CYAN}{'─' * (w - 2)}{_R}")
    print(f"  {_GREEN}{_BOLD}✅  {filename} → queued{_R}")
    print(f"  {_GRAY}    phash  : {phash[:16]}...{_R}")
    print(f"  {_GRAY}    caption: {caption}{_R}")


# ── Summaries ─────────────────────────────────────────────────────────────────

def run_summary(pulled: int, queued_count: int, rejected_by: dict) -> None:
    w = _w()
    total_rejected = sum(rejected_by.values())
    print()
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print(f"{_GREEN}{_BOLD}  ✅  RUN COMPLETE{_R}")
    print(f"{_GREEN}  📥  Pulled     {pulled:>4}{_R}")
    print(f"{_GREEN}  ✅  Queued     {queued_count:>4}{_R}")
    if total_rejected:
        print(f"{_RED}  ✖   Rejected   {total_rejected:>4}{_R}")
        for stage, count in sorted(rejected_by.items(), key=lambda x: -x[1]):
            print(f"{_GRAY}       {stage:<16} {count}{_R}")
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print()


def review_header(index: int, total: int, filename: str,
                  description: str, caption: str, reason: str) -> None:
    w = _w()
    print()
    print(f"{_BOLD}{'━' * w}{_R}")
    print(f"{_BOLD}  🖼   [{index} / {total}]  {filename}{_R}")
    print(f"{'─' * w}")
    print(f"  {_GRAY}📝  {description}{_R}")
    print(f"  {_BOLD}💬  Caption: {caption}{_R}")
    print(f"  {_GRAY}🤔  Gemini:  {reason}{_R}")
    print()


def review_prompt() -> None:
    print(f"  {_CYAN}[y]{_R} post  {_CYAN}[n]{_R} skip  {_CYAN}[e]{_R} edit caption  {_CYAN}[q]{_R} quit", end="  ")


def post_banner(filename: str, caption: str, dry_run: bool) -> None:
    w = _w()
    tag = "  [DRY RUN]" if dry_run else ""
    print(f"  {_CYAN}{'─' * (w - 2)}{_R}")
    print(f"  {_BOLD}📤  POSTING{tag}  {filename}{_R}")
    print(f"  {_GRAY}    Caption: {caption}{_R}")


def post_ok(tweet_url: str) -> None:
    print(f"  {_GREEN}✅  Posted → {tweet_url}{_R}")


def post_dry(filename: str, caption: str) -> None:
    print(f"  {_YELLOW}⚠   [DRY RUN] would post {filename}: {caption}{_R}")


def post_summary(posted: int, dry_run: bool) -> None:
    w = _w()
    tag = "  [DRY RUN]" if dry_run else ""
    print()
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print(f"{_GREEN}{_BOLD}  ✅  POST COMPLETE{tag} — {posted} tweet{'s' if posted != 1 else ''} posted{_R}")
    print(f"{_GREEN}{_BOLD}{'═' * w}{_R}")
    print()
