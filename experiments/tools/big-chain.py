#!/usr/bin/env python3
"""
Big-chain.py - Complete pipeline for CUDA analysis and energy estimation.

Pipeline steps:
1. Inject profiling code into CUDA file using 'injector'
2. Parse injected CUDA to CLang LL IR
3. Parse LL bytecode with LLVM LLI (outputs JSON file with kernel params called "kernel_params.json")
4. Compile original CUDA to PTX using NVCC
5. Analyze PTX using ptx-analyser
6. Print energy estimate
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete CUDA analysis pipeline: "
        "inject -> parse -> compile -> analyze -> estimate"
    )
    parser.add_argument(
        "cuda_file",
        type=Path,
        help="Path to CUDA source file (.cu)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for intermediate files (default: temp directory)",
    )
    parser.add_argument(
        "--csv-kernel-name",
        type=str,
        default=None,
        help="Optional kernel name value to forward to ptx-analyser",
    )
    parser.add_argument(
        "--csv-power-consumption-joules",
        type=str,
        default=None,
        help="Optional power consumption (joules) value to forward to ptx-analyser",
    )
    parser.add_argument(
        "--args",
        dest="lli_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to lli (must be last argument)",
    )
    
    args = parser.parse_args()
    cuda_file = args.cuda_file.resolve()
    
    if not cuda_file.exists():
        print(f"Error: CUDA file not found: {cuda_file}", file=sys.stderr)
        return 1
    
    if not cuda_file.suffix == ".cu":
        print(f"Error: File must have .cu extension: {cuda_file}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {output_dir}")

    # Step 1: Inject CUDA file with headers before running LLVM LLI
    # We pass the file content through stdin to the injector, and capture stdout

    with open(cuda_file, "r") as f:
        cuda_content = f.read()

        injector_cmd = ["injector"]
        print(f"Running injector with file passed to stdin")
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
            print(f"Error running injector: {e.stderr}", file=sys.stderr)
            return 1

        injected_cuda = injector_process.stdout
        injected_cuda_path = output_dir / f"{cuda_file.stem}_injected.cu"
        with open(injected_cuda_path, "w") as f:
            f.write(injected_cuda)
        print(f"Injected CUDA written to: {injected_cuda_path}")

        # Step 2: Parse injected CUDA to CLang LL IR (clang++ -DUSE_LLI -S -emit-llvm modified_kernel.cu --no-cuda-version-check)
        clang_cmd = [
            "clang++",
            "-DUSE_LLI",
            "-S",
            "-emit-llvm",
            "-Iinclude",
            str(injected_cuda_path),
            "--no-cuda-version-check"
        ]
        print(f"Running clang to generate LLVM IR")
        try:
            clang_process = subprocess.run(
                clang_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running clang: {e.stderr}", file=sys.stderr)
            return 1

        # Step 3: ...
        ll_path = f"{cuda_file.stem}_injected.ll"

        lli_cmd = [
            "lli",
            str(ll_path),
            *args.lli_args,
        ]

        print(f"Running LLVM LLI to execute LLVM IR")
        try:
            lli_process = subprocess.run(
                lli_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running LLVM LLI: {e.stderr}", file=sys.stderr)
            return 1

        # Read kernel parameters from JSON output
        kernel_params_path = "kernel_params.json"
        if not Path(kernel_params_path).exists():
            print(f"Error: kernel parameters JSON file not found: {kernel_params_path}", file=sys.stderr)
            return 1

        print(f"Reading kernel parameters from: {kernel_params_path}")
        with open(kernel_params_path, "r") as f:
            kernel_params = json.load(f)

        # Step 4: Compile original CUDA to PTX using NVCC
        nvcc_cmd = [
            "nvcc",
            "-ptx",
            "-Xptxas",
            "-g",
            "-G",
            "-O0",
            "-I", "./include",
            "-arch=sm_89",
            str(cuda_file),
        ]

        print(f"Running NVCC to compile CUDA to PTX")
        try:
            nvcc_process = subprocess.run(
                nvcc_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running NVCC: {e.stderr}", file=sys.stderr)
            return 1

        # Step 5: Analyze PTX using ptx-analyser
        ptx_path = f"{cuda_file.stem}.ptx"
        ptx_analyser_cmd = [
            "ptx-analyser",
            "analyze-cfg",
            "--kernel-params", json.dumps(kernel_params),
        ]

        if args.csv_kernel_name is not None:
            ptx_analyser_cmd.extend(["--csv-kernel-name", args.csv_kernel_name])

        if args.csv_power_consumption_joules is not None:
            ptx_analyser_cmd.extend([
                "--csv-power-consumption-joules",
                args.csv_power_consumption_joules,
            ])

        ptx_analyser_cmd.append(str(ptx_path))

        print(f"Running ptx-analyser to analyze PTX")
        try:
            ptx_analyser_process = subprocess.run(
                ptx_analyser_cmd,
                check=True,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running ptx-analyser: {e.stderr}", file=sys.stderr)
            return 1

if __name__ == "__main__":
    raise SystemExit(main())
