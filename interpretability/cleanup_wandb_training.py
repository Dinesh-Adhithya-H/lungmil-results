#!/usr/bin/env python3
"""Delete multitask training runs from chicago-mil wandb project.

Targets runs whose names start with any of the PREFIXES below.
Pass --dry-run to list without deleting.
"""
import argparse
import wandb

PREFIXES = (
    "longitudinal_mk_mt",
    "set_mil_mt",
    "mario_kempes",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="chicago-mil")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix", nargs="+", default=None,
                        help="Override which prefixes to match (default: all multitask)")
    args = parser.parse_args()

    prefixes = tuple(args.prefix) if args.prefix else PREFIXES
    print(f"Project : {args.project}")
    print(f"Prefixes: {prefixes}")
    print(f"Dry-run : {args.dry_run}\n")

    api = wandb.Api()
    runs = api.runs(args.project)
    matched, deleted = 0, 0
    for run in runs:
        if run.name.startswith(prefixes):
            matched += 1
            print(f"  {'[DRY]' if args.dry_run else 'DEL'} {run.name}  id={run.id}  state={run.state}")
            if not args.dry_run:
                run.delete()
                deleted += 1

    if args.dry_run:
        print(f"\nWould delete {matched} runs (re-run without --dry-run to confirm)")
    else:
        print(f"\nDeleted {deleted} / {matched} matched runs from {args.project}")

if __name__ == "__main__":
    main()
