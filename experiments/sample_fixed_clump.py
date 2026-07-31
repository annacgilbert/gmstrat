"""Emit or run the independent CycleWalk chains in a fixed-clump config.

The default is a dry run.  Pass ``--execute`` to launch missing atlases.
Existing atlases are never overwritten unless ``--force`` is also passed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _chain_command(
    run: dict[str, Any], entry: dict[str, Any], defaults: dict[str, Any]
) -> list[str]:
    if "path" not in entry or "seed" not in entry:
        raise ValueError("each sampling file entry must contain path and seed")
    options = {**defaults, **run.get("sampling", {}), **entry.get("sampling", {})}
    return [
        str(REPO_ROOT / "sampling" / "run.sh"),
        "--map-file",
        str(_resolve(run["graph"])),
        "--output-file",
        str(_resolve(entry["path"])),
        "--pop-dev",
        str(options.get("pop_dev", 0.0)),
        "--gamma",
        "0.0",
        "--iso-weight",
        "0.0",
        "--cycle-walk-steps",
        str(options.get("cycle_walk_steps", "2e6")),
        "--cycle-walk-out-freq",
        str(options.get("cycle_walk_out_freq", 100)),
        "--run-diagnostics",
        str(options.get("run_diagnostics", False)).lower(),
        "--rng-seed",
        str(int(entry["seed"])),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing atlas (never implied by --execute)",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    defaults = config.get("sampling", {})
    seen_seeds: dict[int, set[int]] = {}
    jobs: list[list[str]] = []
    for run in config["runs"]:
        n = int(run["n"])
        seen_seeds[n] = set()
        for group in ("pilot_files", "heldout_files"):
            for entry in run[group]:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "sample_fixed_clump requires {path, seed} file entries"
                    )
                seed = int(entry["seed"])
                if seed in seen_seeds[n]:
                    raise ValueError(f"n={n} reuses RNG seed {seed}")
                seen_seeds[n].add(seed)
                jobs.append(_chain_command(run, entry, defaults))

    for command in jobs:
        output = Path(command[command.index("--output-file") + 1])
        if output.exists() and not args.force:
            print(f"skip existing: {output}")
            continue
        print(shlex.join(command))
        if args.execute:
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
