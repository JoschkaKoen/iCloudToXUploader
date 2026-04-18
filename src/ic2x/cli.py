"""
ic2x — iCloud → X Photo Uploader

Commands:
  ic2x run     Pull from iCloud, filter, dedup, AI checks, queue
  ic2x review  Terminal UI: approve / skip / edit caption / quit
  ic2x post    Post all approved images to X
  ic2x unstick Reset rows stuck in 'posting' back to 'approved'
"""

import argparse
import logging


def main() -> None:
    p = argparse.ArgumentParser(
        prog="ic2x",
        description="iCloud → X photo uploader pipeline",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run",     help="Pull from iCloud and queue photos for review")
    sub.add_parser("review",  help="Approve or reject queued photos")
    sub.add_parser("post",    help="Post approved photos to X")
    sub.add_parser("unstick", help="Reset rows stuck in 'posting' status back to 'approved'")

    args = p.parse_args()

    if args.cmd == "run":
        from ic2x.run import run
        run()
    elif args.cmd == "review":
        from ic2x.review import review
        review()
    elif args.cmd == "post":
        from ic2x.post import post
        post()
    elif args.cmd == "unstick":
        from ic2x.unstick import unstick
        unstick()
