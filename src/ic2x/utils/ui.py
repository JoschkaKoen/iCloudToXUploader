"""
Console UI helpers — pretty terminal output for ic2x.
All functions use rich Console directly so they never appear in log files.
Ported and adapted from XBot-3/utils/ui.py.
"""

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()               # stdout — all UI output
err_console = Console(stderr=True)  # stderr — only ui.err()


# ── Startup ───────────────────────────────────────────────────────────────────

def startup_banner(cfg) -> None:
    """Printed once at the start of ic2x run."""
    from ic2x.utils.ai_client import parse_model_effort
    dry = "YES  ⚠" if cfg.x_dry_run else "no"
    enhance = "enabled" if cfg.enhance_enabled else "disabled"
    judge_model, judge_effort = parse_model_effort(cfg.judge_model)
    judge_str = f"{judge_model} [{judge_effort or 'default'}]"
    rows = [
        ("👤", "iCloud user",   cfg.icloud_username),
        ("🔢", "Recent count",  str(cfg.icloud_recent_count)),
        ("🤖", "Judge model",   judge_str),
        ("✨", "Enhance",       enhance),
        ("🐦", "Dry run",       dry),
    ]
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column(style="cyan")
    for icon, label, value in rows:
        t.add_row(f"{icon}  {label}", escape(value))
    console.print()
    console.print(Panel(
        t,
        title="[bold cyan]📸  IC2X — iCloud → X Photo Pipeline[/]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    console.print()


# ── Section headers ───────────────────────────────────────────────────────────

def run_banner() -> None:
    console.print()
    console.rule("[bold]📥  PULLING FROM iCLOUD[/]", style="bold")
    console.print()


def review_section_banner(total: int) -> None:
    label = f"🖼   REVIEW QUEUE  ({total} image{'s' if total != 1 else ''})"
    console.print()
    console.rule(f"[bold]{escape(label)}[/]", style="bold")
    console.print()


def post_section_banner(total: int, dry_run: bool) -> None:
    tag = "  [DRY RUN]" if dry_run else ""
    label = f"📤  POSTING{tag}  ({total} image{'s' if total != 1 else ''})"
    console.print()
    console.rule(f"[bold]{escape(label)}[/]", style="bold")
    console.print()


# ── Per-image headers ─────────────────────────────────────────────────────────

def file_header(filename: str, index: int, total: int) -> None:
    console.print()
    console.print(f"  [bold]📷  {escape(filename)}[/]  [dim]({index} / {total})[/]")


def stage_banner(step: int, total: int, name: str) -> None:
    console.rule(
        f"[cyan]▶  [{step}/{total}]  {escape(name.upper())}[/]",
        style="cyan",
        align="left",
    )


# ── Inline status ─────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    console.print(f"  [green]✅  {escape(msg)}[/]")


def info(msg: str) -> None:
    console.print(f"  [dim]ℹ   {escape(msg)}[/]")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠   {escape(msg)}[/]")


def err(msg: str) -> None:
    err_console.print(f"  [red]✖   {escape(msg)}[/]")


# ── Decision outputs ──────────────────────────────────────────────────────────

def rejected(filename: str, stage: str, reason) -> None:
    if isinstance(reason, list):
        reason_str = ", ".join(str(r) for r in reason)
    else:
        reason_str = str(reason) if reason else stage
    console.print(
        f"  [red]✖  {escape(filename)} → rejected "
        f"[{escape(stage)}: {escape(reason_str)}][/]"
    )


def queued(filename: str, phash: str, caption: str) -> None:
    console.rule(style="cyan")
    console.print(f"  [green bold]✅  {escape(filename)} → queued[/]")
    console.print(f"  [dim]    phash  : {escape(phash[:16])}...[/]")
    console.print(f"  [dim]    caption: {escape(caption)}[/]")


# ── Summaries ─────────────────────────────────────────────────────────────────

def run_summary(pulled: int, queued_count: int, rejected_by: dict) -> None:
    total_rejected = sum(rejected_by.values())
    t = Table.grid(padding=(0, 2))
    t.add_column(style="green")
    t.add_column(justify="right", style="green")
    t.add_row("📥  Pulled", str(pulled))
    t.add_row("✅  Queued", str(queued_count))
    if total_rejected:
        t.add_row(f"[red]✖   Rejected[/]", f"[red]{total_rejected}[/]")
        for stage, count in sorted(rejected_by.items(), key=lambda x: -x[1]):
            t.add_row(f"[dim]     {escape(stage)}[/]", f"[dim]{count}[/]")
    console.print()
    console.print(Panel(
        t,
        title="[bold green]✅  RUN COMPLETE[/]",
        border_style="green",
        box=box.DOUBLE,
    ))
    console.print()


def review_header(index: int, total: int, filename: str,
                  description: str, caption: str, reason: str) -> None:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("📝", escape(description))
    t.add_row("[bold]💬[/]", f"[bold]Caption:[/] {escape(caption)}")
    t.add_row("🤔", f"[dim]Model:   {escape(reason)}[/]")
    console.print()
    console.print(Panel(
        t,
        title=f"[bold]🖼   [{index} / {total}]  {escape(filename)}[/]",
        border_style="bold",
    ))
    console.print()


def review_prompt() -> None:
    # end="  " keeps the cursor on the same line so input() lands right after
    console.print(
        r"  [cyan]\[y][/] post  [cyan]\[n][/] skip  "
        r"[cyan]\[e][/] edit caption  [cyan]\[q][/] quit",
        end="  ",
    )


def post_banner(filename: str, caption: str, dry_run: bool) -> None:
    tag = "  [DRY RUN]" if dry_run else ""
    console.rule(style="cyan")
    console.print(f"  [bold]📤  POSTING{escape(tag)}  {escape(filename)}[/]")
    console.print(f"  [dim]    Caption: {escape(caption)}[/]")


def post_ok(tweet_url: str) -> None:
    console.print(f"  [green]✅  Posted → {escape(tweet_url)}[/]")


def post_dry(filename: str, caption: str) -> None:
    console.print(
        f"  [yellow]⚠   [DRY RUN] would post {escape(filename)}: {escape(caption)}[/]"
    )


def post_summary(posted: int, dry_run: bool) -> None:
    tag = "  [DRY RUN]" if dry_run else ""
    label = f"✅  POST COMPLETE{tag} — {posted} tweet{'s' if posted != 1 else ''} posted"
    console.print()
    console.print(Panel(
        f"[bold green]{escape(label)}[/]",
        border_style="green",
        box=box.DOUBLE,
    ))
    console.print()
