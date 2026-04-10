#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class RunnerConfig:
    """Configuration for the runner."""
    """cuda: Path to the CUDA source file."""
    cuda: Path
    """run_args: Optional dictionary of preprocessor defines to pass during compilation."""
    run_args: Optional[Dict[str, str]] = None
    """binary_args: Optional list of arguments to pass to the compiled binary at runtime."""
    binary_args: Optional[List[str]] = None


def get_define_flags(run_args: Optional[Dict[str, str]]) -> List[str]:
    if run_args is None:
        return []
    return [f"-D{key}={value}" for key, value in run_args.items()]

def run_command(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def run_command_with_live_output(cmd, cwd=None, stop_on_output_token: Optional[str] = None):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    detected_stop_token = False
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
            if stop_on_output_token and stop_on_output_token in line:
                detected_stop_token = True
                print(
                    f"Detected '{stop_on_output_token}' in program output. Stopping execution.",
                    file=sys.stderr,
                )
                process.terminate()
                break

    if detected_stop_token:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    returncode = process.wait()
    completed = subprocess.CompletedProcess(cmd, returncode, "".join(output_lines))
    completed.detected_stop_token = detected_stop_token
    return completed


def add_tree_to_folder(output_folder: Path, root: Path, base_parent: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative_path = path.relative_to(base_parent)
        destination = output_folder / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def resolve_app_name(cuda_source: Path) -> str:
    parts = cuda_source.parts
    app_index = None
    for index, part in enumerate(parts):
        if part == "apps":
            app_index = index

    if app_index is not None and app_index + 1 < len(parts):
        return parts[app_index + 1]

    folder = cuda_source.parent
    if folder.name == "src" and folder.parent != folder:
        return folder.parent.name

    return folder.name


def run_runner(config: RunnerConfig) -> int:
    """Run madsens jank, does (compilation, execution) 
    
    Args:
        config: RunnerConfig object containing all necessary parameters
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    cuda_source = config.cuda.resolve()
    folder = cuda_source.parent

    if not cuda_source.exists() or not cuda_source.is_file():
        print(f"CUDA source file not found: {cuda_source}", file=sys.stderr)
        return 2

    if cuda_source.suffix != ".cu":
        print(f"Expected a .cu source file, got: {cuda_source}", file=sys.stderr)
        return 2

    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2

    compiler = "nvcc"
    sources = [cuda_source]

    if shutil.which(compiler) is None:
        print(f"Compiler not found in PATH: {compiler}", file=sys.stderr)
        return 2

    build_root = Path("experiments/build").resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    binary_name = resolve_app_name(cuda_source)
    build_dir = build_root / binary_name
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / binary_name
    define_flags = get_define_flags(config.run_args)

    # We run with O0 to ensure we capture the original source.
    include_path = Path(__file__).parents[2] / "include"
    default_args = ["-Xptxas", "-g", "-G", "-O0", f"-I{include_path}", "-arch=sm_89"]

    compile_cmd = [compiler] + default_args + define_flags + ["-o", str(binary_path), "-lcupti"] + [str(p) for p in sources]
    compile_result = run_command(compile_cmd)

    compile_log = build_dir / "compile.log"
    compile_log.write_text(compile_result.stdout)

    if compile_result.returncode != 0:
        print("Compilation failed. See compile.log.")
        print(compile_log)
        return compile_result.returncode

    ptx_files = []
    ptx_log = build_dir / "ptx.log"
    if compiler == "nvcc":
        ptx_outputs = []
        ptx_failed = False
        for src in sources:
            ptx_out = build_dir / f"{src.stem}.ptx"
            ptx_cmd = [compiler, "-ptx"] + default_args + define_flags + ["-o", str(ptx_out), str(src)]
            ptx_result = run_command(ptx_cmd)
            ptx_outputs.append(
                " ".join(ptx_cmd) + "\n" + ptx_result.stdout + "\n"
            )
            if ptx_result.returncode != 0:
                ptx_failed = True
            else:
                ptx_files.append(ptx_out)

        ptx_log.write_text("\n".join(ptx_outputs))
        if ptx_failed:
            print("PTX generation failed. See ptx.log.")
            print(ptx_log)
            return 2

    run_cmd = [str(binary_path)] + (config.binary_args or [])

    # Start nvidia-smi monitoring on a separate process
    monitor_log = build_dir / "nvidia-smi.csv"
    monitor_process = None
    
    # Start pmd2-cli monitoring on a separate process
    pmd2_log = build_dir / "pmd2.csv"
    pmd2_process = None
    if shutil.which("nvidia-smi") is not None:
        try:
            with open(monitor_log, "w") as monitor_file:
                monitor_process = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,fan.speed",
                        "--format=csv",
                        "-lms",
                        "5",
                    ],
                    stdout=monitor_file,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except Exception as e:
            print(f"Warning: Failed to start nvidia-smi monitoring: {e}", file=sys.stderr)
            monitor_process = None
    
    if shutil.which("pmd2-cli") is not None:
        try:
            with open(pmd2_log, "w") as pmd2_file:
                pmd2_process = subprocess.Popen(
                    [
                        "pmd2-cli",
                        "-p",
                        "/dev/ttyACM0",
                        "-c",
                        "-i",
                        "100",
                        "monitor"
                    ],
                    stdout=pmd2_file,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except Exception as e:
            print(f"Warning: Failed to start pmd2-cli monitoring: {e}", file=sys.stderr)
            pmd2_process = None

    start_time = datetime.datetime.now()

    # Give monitors a moment to start
    if monitor_process is not None or pmd2_process is not None:
        time.sleep(1)

    error_marker = "[ERROR]"
    run_result = run_command_with_live_output(run_cmd, stop_on_output_token=error_marker)

    # Surface application-level error markers even when process exit code is 0.
    run_output = run_result.stdout or ""
    has_error_marker = error_marker in run_output or bool(
        getattr(run_result, "detected_stop_token", False)
    )

    # Give monitors a moment to capture final data
    if monitor_process is not None or pmd2_process is not None:
        time.sleep(3)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Stop nvidia-smi monitoring
    if monitor_process is not None:
        monitor_process.terminate()
        try:
            monitor_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor_process.kill()
            monitor_process.wait()
    
    # Stop pmd2-cli monitoring
    if pmd2_process is not None:
        pmd2_process.terminate()
        try:
            pmd2_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pmd2_process.kill()
            pmd2_process.wait()

    output_log = build_dir / "output.txt"
    output_log.write_text(run_result.stdout)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "folder": str(folder),
        "compiler": compiler,
        "compile_command": compile_cmd,
        "run_command": run_cmd,
        "run_returncode": run_result.returncode,
        "stdout_contains_error_marker": has_error_marker,
    }
    if compiler == "nvcc":
        manifest["ptx_files"] = [str(p) for p in ptx_files]
    if monitor_log.exists():
        manifest["monitor_log"] = "nvidia-smi.csv"
    if pmd2_log.exists():
        manifest["pmd2_log"] = "pmd2.csv"
    manifest_path = build_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if has_error_marker:
        print("Run marked as failed due to error marker in stdout.", file=sys.stderr)
        return run_result.returncode if run_result.returncode != 0 else 1
    else:
        print(f"Build and run completed. Output files stored in: {build_dir}")
        print(f"Execution time: {duration:.2f} seconds")
        print("NOTE: Artifact feature removed. Use tools/artifact.py to zip build folder.")
        
    return 0


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Compile, run, capture output, and zip source+binary+output."
    )
    parser.add_argument(
        "cuda",
        help="Path to the CUDA source file to compile and run (e.g., app/main.cu)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to place zip bundles",
    )
    parser.add_argument(
        "--run-args",
        nargs="*",
        help="Preprocessor defines as KEY=VALUE pairs (example: --run-args M=64 N=64 K=64)",
    )
    parser.add_argument(
        "--binary-args",
        nargs="*",
        help="Arguments to pass to the compiled binary at runtime (example: --binary-args 256 256)",
    )

    args = parser.parse_args()

    run_args: Optional[Dict[str, str]] = None
    if args.run_args:
        run_args = {}
        for item in args.run_args:
            if "=" not in item:
                raise ValueError(f"Invalid --run-args entry '{item}'. Expected KEY=VALUE.")
            key, value = item.split("=", 1)
            if not key:
                raise ValueError(f"Invalid --run-args entry '{item}'. Key cannot be empty.")
            run_args[key] = value

    return RunnerConfig(
        cuda=Path(args.cuda),
        run_args=run_args,
        binary_args=args.binary_args,
    )

def main():
    """Main entry point."""
    config = parse_args()
    return run_runner(config)


if __name__ == "__main__":
    raise SystemExit(main())
