#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple

import artifact
import runner
import random

sizes = ["small", "medium", "large", "giant"]
animals = ["cat", "dog", "mouse", "elephant", "giraffe", "lion", "tiger", "bear", "wolf", "fox", "monkey", "panda", "koala", "kangaroo", "zebra", "hippo", "rhino", "deer", "squirrel", "rabbit"]
colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "lime", "teal", "navy", "maroon", "olive", "silver", "gold"]

def random_prefix() -> str:
    return f"{random.choice(sizes)}_{random.choice(colors)}_{random.choice(animals)}"

def _read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def _parse_config_entry(entry: Any, index: int) -> runner.RunnerConfig:
	if not isinstance(entry, dict):
		raise ValueError(f"Entry {index} must be an object.")

	if "cuda" not in entry:
		raise ValueError(f"Entry {index} is missing required field 'cuda'.")

	cuda = Path(entry["cuda"]).resolve()
	if cuda.suffix != ".cu":
		raise ValueError(f"Entry {index} has non-.cu cuda path: {entry['cuda']}")

	run_args_raw = entry.get("run_args")
	if run_args_raw is None:
		run_args = None
	elif isinstance(run_args_raw, list) and all(isinstance(arg, str) for arg in run_args_raw):
		run_args = run_args_raw
	else:
		raise ValueError(f"Entry {index} field 'run_args' must be an array of strings.")

	return runner.RunnerConfig(
		cuda=cuda,
		run_args=run_args,
	)


def load_runner_configs(config_path: Path) -> List[runner.RunnerConfig]:
	data = _read_json(config_path)
	if not isinstance(data, list):
		raise ValueError("Top-level JSON must be an array of RunnerConfig objects.")

	configs: List[runner.RunnerConfig] = []
	for index, entry in enumerate(data):
		configs.append(_parse_config_entry(entry, index))

	if not configs:
		raise ValueError("JSON array is empty. Provide at least one RunnerConfig object.")

	return configs


def _resolve_src_folder(cuda_path: Path) -> Path:
    resolved = cuda_path.resolve()
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "apps" and index + 1 < len(parts):
            apps_root = Path(*parts[: index + 1])
            return apps_root / parts[index + 1]

    if resolved.parent.name == "src" and resolved.parent.parent != resolved.parent:
        return resolved.parent.parent

    return resolved.parent


def _run_artifact_for_config(config: runner.RunnerConfig, artifact_output_dir: Path, prefix: str) -> int:
    app_name = runner.resolve_app_name(config.cuda.resolve())
    build_folder = Path("experiments/build") / app_name
    src_folder = _resolve_src_folder(config.cuda)

    artifact_config = artifact.ArtifactConfig(
        build_folder=build_folder,
        src_folder=src_folder,
        output_dir=artifact_output_dir,
        prefix=prefix
    )
    return artifact.run_artifact(artifact_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple RunnerConfig entries from a JSON file in sequence"
    )
    parser.add_argument(
        "config_json",
        help="Path to JSON file containing an array of RunnerConfig objects",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed run",
    )
    parser.add_argument(
        "--artifact-output-dir",
        default="experiments/artifacts",
        help="Directory where artifact bundles are created after each run",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_json)
    if not config_path.exists() or not config_path.is_file():
        print(f"Config JSON not found: {config_path}")
        return 2

    try:
        configs = load_runner_configs(config_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to parse config JSON: {exc}")
        return 2

    artifact_output_dir = Path(args.artifact_output_dir)
    results: List[Tuple[Path, int, int]] = []
    prefix = random_prefix()
    for config in configs:
        print(f"\n=== Running: (prefix: {prefix}) {config.cuda} ===")
        code = runner.run_runner(config)

        print(f"=== Creating artifact for: {config.cuda} ===")
        artifact_code = _run_artifact_for_config(config, artifact_output_dir, prefix)
        results.append((config.cuda, code, artifact_code))

        if code != 0 and args.fail_fast:
            break

        print(
            f"\n=== Finished: {config.cuda} run={code}, artifact={artifact_code} ==="
        )

    print("\n=== Summary ===")
    for cuda_source, run_code, artifact_code in results:
        run_status = "OK" if run_code == 0 else f"FAILED ({run_code})"
        artifact_status = "OK" if artifact_code == 0 else f"FAILED ({artifact_code})"
        print(f"{cuda_source}: run={run_status}, artifact={artifact_status}")

    failed = [1 for _, run_code, artifact_code in results if run_code != 0 or artifact_code != 0]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())