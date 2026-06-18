"""
ic2x — autonomous iCloud → X "best-of-burst" bot.

Commands:
  ic2x            Run the loop (default; same as `ic2x bot`)
  ic2x bot        Fetch newest unseen burst → pick the best → post; every N hours
  ic2x login      Interactive iCloud sign-in (2FA) to establish the session
  ic2x compare    Compare judge models on recent bursts (read-only, no posting)
  ic2x clean      Discard non-posted image records
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="ic2x", description="Autonomous iCloud → X best-of-burst bot")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("bot", help="Run the autonomous loop (default)")
    sub.add_parser("login", help="Interactive iCloud sign-in (2FA) to establish the session")
    cmp = sub.add_parser("compare", help="Compare judge models on recent bursts (read-only)")
    cmp.add_argument("--models", default="gemini-2.5-flash-lite,qwen3.5-flash",
                     help="comma-separated model ids to compare")
    cmp.add_argument("--bursts", type=int, default=5, help="number of recent bursts to test")
    sub.add_parser("clean", help="Discard non-posted image records")

    args = p.parse_args()
    cmd = args.cmd or "bot"

    if cmd == "bot":
        from ic2x.bot import bot
        bot()
    elif cmd == "login":
        from ic2x.login import login
        login()
    elif cmd == "compare":
        from ic2x.compare import compare
        compare(models=[m.strip() for m in args.models.split(",") if m.strip()],
                n_bursts=args.bursts)
    elif cmd == "clean":
        from ic2x.clean import clean
        clean()
