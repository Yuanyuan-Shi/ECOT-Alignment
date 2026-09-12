"""Create the persistent W&B identity before a queued cluster job starts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "wandb.json"
    if identity_path.exists():
        print("WANDB_URL " + json.loads(identity_path.read_text())["url"])
        return
    config = json.loads(os.path.expandvars(Path(args.config).read_text()))
    (root / "wandb").mkdir(exist_ok=True)
    run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                     name=config["wandb_run_name"], config=config, dir=str(root / "wandb"))
    identity = {"id": run.id, "url": run.url}
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    print("WANDB_URL " + run.url, flush=True)
    run.finish()


if __name__ == "__main__":
    main()
