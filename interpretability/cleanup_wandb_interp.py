#!/usr/bin/env python3
"""Delete all runs in chicago-mil-interpretability wandb project."""
import wandb

api = wandb.Api()
project = "chicago-mil-interpretability"
entity  = None   # uses default from ~/.netrc

runs = api.runs(f"{entity}/{project}" if entity else project)
deleted = 0
for run in runs:
    print(f"  deleting: {run.name} ({run.id}) group={run.group}")
    run.delete()
    deleted += 1

print(f"\nDeleted {deleted} runs from {project}")
