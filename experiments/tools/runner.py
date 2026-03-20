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
from typing import Optional, List


@dataclass
class RunnerConfig:
    """Configuration for the runner."""
    """cuda: Path to the CUDA source file."""
    cuda: Path
    """output_dir: Path to the directory where output artifacts will be stored."""
    output_dir: Path
    """run_args: Optional list of arguments to pass to the compiled binary when running."""
    run_args: Optional[List[str]] = None


def find_sources(root: Path):
    cu_files = list(root.rglob("*.cu"))
    if cu_files:
        return "nvcc", cu_files

    cpp_files = []
    for ext in ("*.cpp", "*.cc", "*.cxx"):
        cpp_files.extend(root.rglob(ext))
    if cpp_files:
        return "g++", cpp_files

    c_files = list(root.rglob("*.c"))
    if c_files:
        return "gcc", c_files

    return None, []


def run_command(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


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

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    build_root = Path("experiments/build").resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    binary_name = resolve_app_name(cuda_source)
    build_dir = build_root / binary_name
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / binary_name

    # We run with O0 to ensure we capture the original source.
    default_args = ["-Xptxas", "-g", "-G", "-O0", "-I", "./experiments/include", "-arch=sm_89"]

    compile_cmd = [compiler] + default_args + ["-o", str(binary_path), "-lcupti"] + [str(p) for p in sources]
    compile_result = run_command(compile_cmd)

    compile_log = build_dir / "compile.log"
    compile_log.write_text(compile_result.stdout)

    if compile_result.returncode != 0:
        print("Compilation failed. See compile.log.")
        return compile_result.returncode

    ptx_files = []
    ptx_log = build_dir / "ptx.log"
    if compiler == "nvcc":
        ptx_outputs = []
        ptx_failed = False
        for src in sources:
            ptx_out = build_dir / f"{src.stem}.ptx"
            ptx_cmd = [compiler, "-ptx"] + default_args + ["-o", str(ptx_out), str(src)]
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
            return 2

    run_cmd = [str(binary_path)]
    if config.run_args:
        run_cmd += config.run_args

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

    run_result = run_command(run_cmd)

    # Give monitors a moment to capture final data
    if monitor_process is not None or pmd2_process is not None:
        time.sleep(3)

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Execution time: {duration:.2f} seconds")

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
    }
    if compiler == "nvcc":
        manifest["ptx_files"] = [str(p) for p in ptx_files]
    if monitor_log.exists():
        manifest["monitor_log"] = "nvidia-smi.csv"
    if pmd2_log.exists():
        manifest["pmd2_log"] = "pmd2.csv"
    manifest_path = build_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Build and run completed. Output files stored in: {build_dir}")

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
        nargs=argparse.REMAINDER,
        help="Arguments passed to the compiled binary",
    )

    args = parser.parse_args()

    return RunnerConfig(
        cuda=Path(args.cuda),
        output_dir=Path(args.output_dir),
        run_args=args.run_args,
    )

def main():
    """Main entry point."""
    config = parse_args()
    return run_runner(config)


if __name__ == "__main__":
    raise SystemExit(main())
