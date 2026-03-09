#!/usr/bin/env python3
"""
Big-chain.py - Complete pipeline for CUDA analysis and energy estimation.

Pipeline steps:
1. Inject profiling code into CUDA file using 'injector'
2. Parse injected CUDA to CLang LL IR
3. Parse LL bytecode with LLVM LLI (outputs JSON kernel params)
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


class PipelineError(Exception):
    """Raised when a pipeline step fails."""
    pass


def run_command(cmd: list, cwd: Optional[Path] = None, capture_output: bool = False) -> str:
    """
    Run a shell command and return stdout.
    
    Args:
        cmd: Command and arguments as list
        cwd: Working directory
        capture_output: Whether to capture output
        
    Returns:
        stdout as string
        
    Raises:
        PipelineError: If command fails
    """
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise PipelineError(
                f"Command failed with code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        return result.stdout
    except FileNotFoundError as e:
        raise PipelineError(f"Command not found: {cmd[0]}") from e


def step1_inject_profiling(cuda_file: Path, work_dir: Path) -> Path:
    """
    Step 1: Inject profiling code using inspector/injector.
    
    Args:
        cuda_file: Path to original CUDA file
        work_dir: Working directory for outputs
        
    Returns:
        Path to injected CUDA file
    """
    print("\n" + "="*60)
    print("STEP 1: Injecting profiling code with inspector/injector")
    print("="*60)
    
    # Build injector if needed
    inspector_build = Path("inspector/build")
    injector_binary = inspector_build / "injector"
    
    if not injector_binary.exists():
        print(f"Building injector at {inspector_build}...")
        run_command(["cmake", ".."], cwd=inspector_build)
        run_command(["cmake", "--build", "."], cwd=inspector_build)
    
    # Run injector
    injected_cuda = work_dir / f"{cuda_file.stem}_injected.cu"
    with open(injected_cuda, "w") as f:
        run_command([str(injector_binary), str(cuda_file)], capture_output=True)
        f.write(run_command([str(injector_binary), str(cuda_file)]))
    
    print(f"✓ Injected CUDA written to: {injected_cuda}")
    return injected_cuda


def step2_parse_to_llvm_ir(injected_cuda: Path, work_dir: Path) -> Path:
    """
    Step 2: Parse injected CUDA to LLVM IR using clang++.
    
    Args:
        injected_cuda: Path to injected CUDA file
        work_dir: Working directory for outputs
        
    Returns:
        Path to generated .ll file
    """
    print("\n" + "="*60)
    print("STEP 2: Parsing to LLVM IR with clang++")
    print("="*60)
    
    ll_file = work_dir / f"{injected_cuda.stem}.ll"
    
    cmd = [
        "clang++",
        "-DUSE_LLI",
        "-S",
        "-emit-llvm",
        str(injected_cuda),
        "--no-cuda-version-check",
        "-o",
        str(ll_file),
    ]
    
    run_command(cmd)
    print(f"✓ LLVM IR written to: {ll_file}")
    return ll_file


def step3_execute_lli_for_kernel_params(ll_file: Path) -> Dict[str, Any]:
    """
    Step 3: Execute LLVM LLI interpreter to extract kernel parameters.
    
    Args:
        ll_file: Path to LLVM IR file
        
    Returns:
        Kernel parameters as dictionary
    """
    print("\n" + "="*60)
    print("STEP 3: Executing LLVM LLI to extract kernel parameters")
    print("="*60)
    
    cmd = ["lli", str(ll_file)]
    output = run_command(cmd)
    
    print(f"LLI output:\n{output}")
    
    # Try to parse JSON from output
    kernel_params = None
    for line in output.split("\n"):
        try:
            # Look for JSON in the output
            if line.strip().startswith("{"):
                kernel_params = json.loads(line)
                break
        except json.JSONDecodeError:
            continue
    
    if kernel_params is None:
        # If no JSON found, use default parameters
        print("⚠ No kernel parameters found in LLI output, using defaults")
        kernel_params = {
            "gridDim": {"x": 1, "y": 1, "z": 1},
            "blockDim": {"x": 32, "y": 1, "z": 1},
            "parameters": []
        }
    
    print(f"✓ Kernel parameters extracted: {json.dumps(kernel_params, indent=2)}")
    return kernel_params


def step4_compile_to_ptx(cuda_file: Path, work_dir: Path) -> Path:
    """
    Step 4: Compile original CUDA file to PTX using NVCC.
    
    Args:
        cuda_file: Path to original CUDA file
        work_dir: Working directory for outputs
        
    Returns:
        Path to generated .ptx file
    """
    print("\n" + "="*60)
    print("STEP 4: Compiling to PTX with NVCC")
    print("="*60)
    
    ptx_file = work_dir / f"{cuda_file.stem}.ptx"
    
    cmd = [
        "nvcc",
        "-ptx",
        "-Xptxas", "-g",
        "-G",
        "-O0",
        "-arch=sm_89",
        "-o", str(ptx_file),
        str(cuda_file),
    ]
    
    run_command(cmd)
    print(f"✓ PTX written to: {ptx_file}")
    return ptx_file


def step5_analyze_ptx(ptx_file: Path, kernel_params: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """
    Step 5: Analyze PTX using ptx-analyser.
    
    Args:
        ptx_file: Path to PTX file
        kernel_params: Kernel parameters dictionary
        work_dir: Working directory for outputs
        
    Returns:
        Analysis results as dictionary
    """
    print("\n" + "="*60)
    print("STEP 5: Analyzing PTX with ptx-analyser")
    print("="*60)
    
    # First, try to build ptx-analyser if needed
    analyser_dir = Path("ptx-analyser")
    if analyser_dir.exists():
        print("Building ptx-analyser...")
        try:
            run_command(["cargo", "build", "--release"], cwd=analyser_dir)
        except PipelineError:
            print("⚠ ptx-analyser build failed, trying existing binary...")
    
    # Try different possible locations for the binary
    possible_binaries = [
        analyser_dir / "target" / "release" / "ptx-analyser",
        analyser_dir / "target" / "debug" / "ptx-analyser",
        Path("ptx-analyser"),
    ]
    
    analyser_binary = None
    for binary_path in possible_binaries:
        if binary_path.exists():
            analyser_binary = binary_path
            break
    
    if analyser_binary is None:
        raise PipelineError("ptx-analyser binary not found. Please build it.")
    
    # Convert kernel params to JSON string for CLI
    kernel_params_json = json.dumps(kernel_params)
    
    # Run analysis with kernel parameters
    html_output = work_dir / "cfg.html"
    cmd = [
        str(analyser_binary),
        "analyze-cfg",
        "--kernel-params", kernel_params_json,
        str(ptx_file),
    ]
    
    output = run_command(cmd)
    print(f"Analysis output:\n{output}")
    
    # Also generate CFG visualization
    cfg_cmd = [
        str(analyser_binary),
        "build-cfg",
        "--html-output", str(html_output),
        str(ptx_file),
    ]
    
    try:
        run_command(cfg_cmd)
        print(f"✓ CFG visualization written to: {html_output}")
    except PipelineError as e:
        print(f"⚠ CFG generation failed: {e}")
    
    # Parse analysis output
    analysis_result = {
        "ptx_file": str(ptx_file),
        "kernel_params": kernel_params,
        "output": output,
    }
    
    return analysis_result


def step6_estimate_energy(analysis_result: Dict[str, Any], work_dir: Path) -> None:
    """
    Step 6: Print energy estimate based on analysis.
    
    Args:
        analysis_result: Analysis results from ptx-analyser
        work_dir: Working directory
    """
    print("\n" + "="*60)
    print("STEP 6: Energy Estimate")
    print("="*60)
    
    print(f"\nKernel Parameters:")
    params = analysis_result.get("kernel_params", {})
    print(json.dumps(params, indent=2))
    
    print(f"\nPTX Analysis Output:")
    print(analysis_result.get("output", "No output available"))
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


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
    
    args = parser.parse_args()
    cuda_file = args.cuda_file.resolve()
    
    if not cuda_file.exists():
        print(f"Error: CUDA file not found: {cuda_file}", file=sys.stderr)
        return 1
    
    if not cuda_file.suffix == ".cu":
        print(f"Error: File must have .cu extension: {cuda_file}", file=sys.stderr)
        return 1
    
    # Create work directory
    if args.output_dir:
        work_dir = args.output_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="cuda-analysis-"))
    
    print(f"\nCUDA Analysis Pipeline")
    print(f"Input file: {cuda_file}")
    print(f"Work directory: {work_dir}")
    
    try:
        # Step 1: Inject profiling
        injected_cuda = step1_inject_profiling(cuda_file, work_dir)
        
        # Step 2: Parse to LLVM IR
        ll_file = step2_parse_to_llvm_ir(injected_cuda, work_dir)
        
        # Step 3: Execute LLI for kernel params
        kernel_params = step3_execute_lli_for_kernel_params(ll_file)
        
        # Step 4: Compile original to PTX
        ptx_file = step4_compile_to_ptx(cuda_file, work_dir)
        
        # Step 5: Analyze PTX
        analysis_result = step5_analyze_ptx(ptx_file, kernel_params, work_dir)
        
        # Step 6: Print energy estimate
        step6_estimate_energy(analysis_result, work_dir)
        
        return 0
        
    except PipelineError as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⚠ Pipeline interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
