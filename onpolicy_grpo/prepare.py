"""Freeze an isolated GRPO smoke-run configuration and iteration manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from onpolicy_grpo.common import PACKAGE, make_iteration_manifest, read, sha256, validate_manifest, write


def prepare(config_path: str | Path, run_dir_override: str | None = None) -> Path:
    source = Path(config_path).resolve()
    config = json.loads(os.path.expandvars(source.read_text()))
    run_dir = Path(run_dir_override or os.path.expandvars(config["run_dir"])).resolve()
    config["run_dir"] = str(run_dir)
    config["base_checkpoint"] = str(Path(os.path.expandvars(config["base_checkpoint"])).resolve())
    if config["iterations"] != 1:
        raise ValueError("workstation smoke configuration must contain exactly one iteration")
    expected = read(PACKAGE / "selection.json")
    if config["task_ids"] != expected["selected"]:
        raise ValueError("task selection differs from the frozen PPO ten-task subset")
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.json"
    if frozen.exists() and read(frozen) != config:
        raise RuntimeError("refusing to alter an existing frozen smoke configuration")
    write(frozen, config)
    (run_dir / "selection.json").write_text((PACKAGE / "selection.json").read_text())
    manifest = make_iteration_manifest(config, 1)
    validate_manifest(manifest, config["task_ids"], config["group_size"])
    write(run_dir / "iteration-0001.manifest.json", manifest)
    write(run_dir / "provenance.json", {
        "config_source": str(source),
        "base_checkpoint_sha256": sha256(config["base_checkpoint"]),
        "ppo_selection_source": "runs/on-policy-ppo/selection.json",
        "ppo_artifacts_modified": False,
    })
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE / "config_smoke.json"))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    print(prepare(args.config, args.run_dir))


if __name__ == "__main__":
    main()
