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

    botp = sub.add_parser("bot", help="Run the autonomous loop (default)")
    botp.add_argument("--once", action="store_true",
                      help="run a single cycle and exit (for testing; forces a cycle, "
                           "ignores the post interval + daily cap)")
    sub.add_parser("login", help="Interactive iCloud sign-in (2FA) to establish the session")
    cmp = sub.add_parser("compare", help="Compare judge models on recent bursts (read-only)")
    cmp.add_argument("--models", default="gemini-2.5-flash-lite; qwen3.5-flash",
                     help="';'-separated model ids (each may carry a ', effort' suffix, "
                          "e.g. 'qwen3.5-flash, off')")
    cmp.add_argument("--bursts", type=int, default=5, help="number of recent bursts to test")
    cmp.add_argument("--keep", action="store_true",
                     help="save evaluated thumbnails + verdicts.md to compare_out/ for review")
    cls = sub.add_parser("classify",
                         help="Judge N newest non-screenshot photos → classify_out/ (no posting)")
    cls.add_argument("--count", type=int, default=30, help="number of photos to classify")
    cls.add_argument("--rotate", action="store_true",
                     help="run the AI rotation pass on each image before judging")
    rot = sub.add_parser("autorotate",
                         help="Test the AI rotation pass on N newest photos → rotate_out/")
    rot.add_argument("--count", type=int, default=30, help="number of photos to rotate")
    sub.add_parser("clean", help="Discard non-posted image records")

    args = p.parse_args()
    cmd = args.cmd or "bot"

    if cmd == "bot":
        from ic2x.bot import bot
        bot(once=getattr(args, "once", False))
    elif cmd == "login":
        from ic2x.login import login
        login()
    elif cmd == "compare":
        from ic2x.compare import compare
        compare(models=[m.strip() for m in args.models.split(";") if m.strip()],
                n_bursts=args.bursts, keep=args.keep)
    elif cmd == "classify":
        from ic2x.classify import classify
        classify(count=args.count, rotate=args.rotate)
    elif cmd == "autorotate":
        from ic2x.rotate import autorotate
        autorotate(count=args.count)
    elif cmd == "clean":
        from ic2x.clean import clean
        clean()
