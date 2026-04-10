#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
from dataclasses import dataclass
import subprocess
import re

import artifact
import runner
import sys
import random
import power

sizes = ["small", "medium", "large", "giant"]
animals = ["cat", "dog", "mouse", "elephant", "giraffe", "lion", "tiger", "bear", "wolf", "fox", "monkey", "panda", "koala", "kangaroo", "zebra", "hippo", "rhino", "deer", "squirrel", "rabbit"]
colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "lime", "teal", "navy", "maroon", "olive", "silver", "gold"]

@dataclass
class BentrResult:
    cuda_source: Path
    run_args: Optional[Dict[str, str]]
    artifact_path: Path
    run_code: int
    artifact_code: int
    total_energy_j: float
    total_estimated_energy_j: float

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
    elif isinstance(run_args_raw, dict) and all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool))
        for k, v in run_args_raw.items()
    ):
        run_args = {k: str(v) for k, v in run_args_raw.items()}
    else:
        raise ValueError(
            f"Entry {index} field 'run_args' must be an object with string keys and scalar values (string/number/bool)."
        )

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

def _get_define_flags(run_args: Optional[Dict[str, str]]) -> List[str]:
    """Convert run_args dictionary to -D compiler flags."""
    if run_args is None:
        return []
    return [f"-D{key}={value}" for key, value in run_args.items()]



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



def _write_summary_csv(results: List[BentrResult], prefix: str) -> Path:
    summary_path = Path(f"{prefix}.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["cuda_source", "run_args", "artifact_path", "run_code", "artifact_code", "total_energy_j", "total_estimated_energy_j"])
        for result in results:
            writer.writerow([
                str(result.cuda_source),
                json.dumps(result.run_args) if result.run_args is not None else "",
                str(result.artifact_path),
                result.run_code,
                result.artifact_code,
                result.total_energy_j,
                result.total_estimated_energy_j,
            ])
    return summary_path


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
        "--cooldown-seconds",
        type=float,
        default=10,
        help="Seconds to wait between runs (default: 10)",
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
    results: List[BentrResult] = []
    prefix = random_prefix()
    for index, config in enumerate(configs):
        print(f"\n=== Running: (prefix: {prefix}) {config.cuda} ===")
        run_cfg = runner.RunnerConfig(
            cuda=config.cuda,
            run_args=config.run_args.copy() if config.run_args is not None else None,
        )
        code = runner.run_runner(run_cfg)

        print(f"=== Running power estimator ===")
        power_cfg = power.PowerConfig(
            path=Path("experiments/build") / runner.resolve_app_name(config.cuda),
        )
        power_metrics = power.extract_power_metrics(power_cfg)
        total_energy_j = power_metrics.total_energy_j

        print(f"[DEBUG] Power metrics: {power_metrics}")

        print(f"=== Running power prediction ===")

        # Create a temporary directory for intermediate files
        tmp_directory = Path("/tmp/bentr")
        tmp_directory.mkdir(parents=True, exist_ok=True)

        # Step 1: Inject CUDA file with headers before running LLVM LLI
        # We pass the file content through stdin to the injector, and capture stdout

        with open(config.cuda, "r") as f:
            cuda_content = f.read()

        injector_cmd = ["injector"]
        print(f"[DEBUG] Running injector with file passed to stdin")
        try:
            injector_process = subprocess.run(
                injector_cmd,
                input=cuda_content,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[DEBUG] Error running injector: {e.stderr}", file=sys.stderr)
            return 1

        injected_cuda = injector_process.stdout
        injected_cuda_path = tmp_directory / f"{Path(config.cuda).stem}_injected.cu"
        with open(injected_cuda_path, "w") as f:
            f.write(injected_cuda)
        print(f"[DEBUG] Injected CUDA written to: {injected_cuda_path}")

        # Step 2: Parse injected CUDA to Clang LLVM IR and keep outputs in tmp
        ll_path = tmp_directory / f"{Path(config.cuda).stem}_injected.ll"
        bc_path = tmp_directory / f"{Path(config.cuda).stem}_injected.bc"

        clang_ll_cmd = [
            "clang++",
            "-DUSE_LLI",
            "-S",
            "-emit-llvm",
            "--cuda-host-only",
            "-Iinclude",
            "-o",
            str(ll_path),
            str(injected_cuda_path),
            "--no-cuda-version-check"
        ]

        clang_bc_cmd = [
            "clang++",
            "-DUSE_LLI",
            "-emit-llvm",
            "-c",
            "--cuda-host-only",
            "-Iinclude",
            "-o",
            str(bc_path),
            str(injected_cuda_path),
            "--no-cuda-version-check"
        ]
        
        # Add run_args as -D defines
        define_flags = _get_define_flags(config.run_args)
        clang_ll_cmd.extend(define_flags)
        clang_bc_cmd.extend(define_flags)
        
        print(f"[DEBUG] Running clang to generate LLVM IR (.ll and .bc)")

        try:
            subprocess.run(
                clang_ll_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            subprocess.run(
                clang_bc_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] running clang: {e.stderr}", file=sys.stderr)
            return 1

        # Step 3: Run LLVM LLI with injected CUDA and capture JSON output with kernel parameters

        lli_cmd = [
            "lli",
            str(ll_path),
        ]

        print(f"[DEBUG] Running LLVM LLI to execute LLVM IR")
        try:
            subprocess.run(
                lli_cmd,
                check=True,
                cwd=tmp_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error running LLVM LLI: {e.stderr}", file=sys.stderr)
            return 1



        # Read kernel parameters from JSON output
        kernel_params_path = tmp_directory / "kernel_params.json"
        if not kernel_params_path.exists():
            print(f"Error: kernel parameters JSON file not found: {kernel_params_path}", file=sys.stderr)
            return 1

        print(f"Reading kernel parameters from: {kernel_params_path}")
        with kernel_params_path.open("r", encoding="utf-8") as f:
            kernel_params = json.load(f)

        # Step 4: Compile original CUDA to PTX using NVCC
        ptx_path = tmp_directory / f"{Path(config.cuda).stem}.ptx"
        nvcc_cmd = [
            "nvcc",
            "-ptx",
            "-Xptxas",
            "-g",
            "-G",
            "-O0",
            "-I", "./include",
            "-arch=sm_89",
            "-o",
            ptx_path,
            str(config.cuda),
        ]
        
        # Add run_args as -D defines to nvcc as well
        nvcc_cmd.extend(define_flags)

        print(f"Running NVCC to compile CUDA to PTX")
        try:
            subprocess.run(
                nvcc_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error running NVCC: {e.stderr}", file=sys.stderr)
            return 1

        # Step 5: Analyze PTX using ptx-analyser
        ptx_analyser_output = tmp_directory / f"ptx_analysis.json"
        ptx_analyser_cmd = [
            "ptx-analyser",
            "analyze-cfg",
            "--kernel-params", json.dumps(kernel_params),
            "--output-json-path", ptx_analyser_output,
            str(ptx_path)
        ]

        print(f"[DEBUG] Running ptx-analyser to analyze PTX")
        try:
            subprocess.run(
                ptx_analyser_cmd,
                check=True,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error running ptx-analyser: {e.stderr}", file=sys.stderr)
            return 1
        
        print(f"[DEBUG] Running linear model to predict power from PTX analysis")

        linear_model_cmd = [
            "uv",
            "run",
            "./linear-model/linear-model.py",
            "--weights-path",
            "./linear-model/weights.csv",
            "--input-path",
            ptx_analyser_output
        ]
        
        total_estimated_energy_j = None
        try:
            linear_model_process = subprocess.run(
                linear_model_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Extract the '[TOTAL] 0.027659098837 J' line from the output to get the total estimated energy using RegEx
            # We can also capture the entire output for debugging purposes
            linear_model_output = linear_model_process.stdout or ""
            print(f"[DEBUG] Linear model output:\n{linear_model_output}")
            total_estimated_energy_match = re.search(r"\[TOTAL\]\s+([0-9.]+)\s+J", linear_model_output)
            if total_estimated_energy_match:
                total_estimated_energy_j = float(total_estimated_energy_match.group(1))
                print(f"[DEBUG] Extracted total estimated energy: {total_estimated_energy_j:.12f} J")
            else:
                print(f"[ERROR] Failed to extract total estimated energy from linear model output", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error running linear model: {e.stderr}", file=sys.stderr)
            return 1

        if total_estimated_energy_j is None:
            print(f"[ERROR] Total estimated energy not found in linear model output", file=sys.stderr)
            return 1


        print(f"=== Creating artifact for: {config.cuda} ===")
        
        artifact_result = _run_artifact_for_config(config, artifact_output_dir, prefix)

        # Save results for summary at the end
        results.append(BentrResult(
            config.cuda,
            config.run_args.copy() if config.run_args is not None else None,
            artifact_result.output_dir / f"{prefix}_{config.cuda.stem}_artifact.zip",
            code,
            artifact_result.code,
            total_energy_j,
            total_estimated_energy_j
        ))

        if code != 0 and args.fail_fast:
            break

        is_last_run = index == len(configs) - 1
        if not is_last_run and args.cooldown_seconds > 0:
            print(f"=== Cooling down for {args.cooldown_seconds} seconds ===")
            time.sleep(args.cooldown_seconds)


    print("\n=== Summary ===")

    _write_summary_csv(results, prefix)

    print(f"Summary CSV written to: {prefix}.csv")

if __name__ == "__main__":
    raise SystemExit(main())