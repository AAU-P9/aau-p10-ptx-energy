#!/usr/bin/env python3
"""
analyze_ir.py — Extract parameter range constraints from LLVM IR assume metadata.

Parses the device LLVM IR (.ll) file and reconstructs what ranges/constraints
__builtin_assume() calls impose on kernel parameters.

The key insight: llvm.assume() is an LLVM-IR-only construct.  It does NOT
survive into PTX.  So any analysis of parameter ranges MUST happen at the
IR level (this script), not by inspecting PTX.

Usage:
    python3 analyze_ir.py build/device.ll
    python3 analyze_ir.py build/device_opt.ll
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParamConstraint:
    """A single constraint on a kernel parameter."""
    kind: str           # 'range_lo', 'range_hi', 'multiple_of', 'aligned'
    value: int
    raw_ir: str = ""    # the IR line(s) that produced this

@dataclass
class ParamInfo:
    name: str           # e.g. '%3' or 'M'
    ir_reg: str         # e.g. '%3'
    param_index: int    # 0-based index in kernel signature
    llvm_type: str      # e.g. 'i32', 'ptr'
    constraints: list = field(default_factory=list)
    is_noalias: bool = False
    is_nonnull: bool = False
    alignment: Optional[int] = None

    def range_lo(self):
        vals = [c.value for c in self.constraints if c.kind == 'range_lo']
        return max(vals) if vals else None

    def range_hi(self):
        vals = [c.value for c in self.constraints if c.kind == 'range_hi']
        return min(vals) if vals else None

    def multiples(self):
        return [c.value for c in self.constraints if c.kind == 'multiple_of']

    def summary(self):
        parts = []
        lo, hi = self.range_lo(), self.range_hi()
        if lo is not None and hi is not None:
            parts.append(f"range: [{lo}, {hi}]")
        elif lo is not None:
            parts.append(f"min: {lo}")
        elif hi is not None:
            parts.append(f"max: {hi}")
        for m in self.multiples():
            parts.append(f"multiple of {m}")
        if self.is_noalias:
            parts.append("noalias (__restrict__)")
        if self.alignment:
            parts.append(f"aligned to {self.alignment} bytes")
        return ", ".join(parts) if parts else "unconstrained"


def parse_kernel_signature(lines):
    """Find kernel entry and parse parameter types/attributes."""
    kernels = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match: define ... @kernel_name(...)
        m = re.match(r'define\s+.*\s+@(\w+)\((.*)', line)
        if m:
            kernel_name = m.group(1)
            # Collect full signature (may span multiple lines)
            sig = m.group(2)
            while ')' not in sig and i + 1 < len(lines):
                i += 1
                sig += lines[i]

            # Parse parameters
            params = []
            param_idx = 0
            for part in sig.split(','):
                part = part.strip()
                if not part or part.startswith(')'):
                    break

                is_noalias = 'noalias' in part
                ir_type = 'ptr' if 'ptr' in part else 'i32' if 'i32' in part else 'i64' if 'i64' in part else 'unknown'

                # Extract register name (e.g., %0, %1, ...)
                reg_match = re.search(r'(%\d+)', part)
                reg = reg_match.group(1) if reg_match else f'%{param_idx}'

                p = ParamInfo(
                    name=reg,
                    ir_reg=reg,
                    param_index=param_idx,
                    llvm_type=ir_type,
                    is_noalias=is_noalias,
                )
                params.append(p)
                param_idx += 1

            kernels[kernel_name] = params
        i += 1
    return kernels


def trace_assume_constraints(lines, params_by_reg):
    """
    Walk the IR and trace llvm.assume(i1 %cond) calls back to their
    icmp/and instructions to extract range constraints on parameters.

    Patterns recognised:
      - icmp uge/ugt/sge/sgt %param, <const>  -> range_lo
      - icmp ule/ult/sle/slt %param, <const>  -> range_hi
      - icmp ne %param, 0                     -> range_lo >= 1
      - and %param, <mask>; icmp eq %, 0      -> multiple_of (mask+1)
      - ptrtoint ptr %param; and %, <mask>; icmp eq %, 0 -> aligned (mask+1)
    """
    # Build a map of %register -> defining instruction
    defs = {}
    for line in lines:
        line = line.strip()
        m = re.match(r'(%\d+)\s*=\s*(.*)', line)
        if m:
            defs[m.group(1)] = m.group(2)

    # Find all assume calls
    for line in lines:
        line = line.strip()
        assume_match = re.search(r'call void @llvm\.assume\(i1\s+(%\d+)\)', line)
        if not assume_match:
            continue

        cond_reg = assume_match.group(1)
        if cond_reg not in defs:
            continue

        cond_def = defs[cond_reg]

        # Pattern: icmp <pred> <type> %src, <const>
        icmp_match = re.match(
            r'icmp\s+(eq|ne|ult|ule|ugt|uge|slt|sle|sgt|sge)\s+\w+\s+(%\d+),\s*(-?\d+)',
            cond_def
        )
        if icmp_match:
            pred, src_reg, const_val = icmp_match.group(1), icmp_match.group(2), int(icmp_match.group(3))

            # Direct parameter reference
            if src_reg in params_by_reg:
                param = params_by_reg[src_reg]
                _add_icmp_constraint(param, pred, const_val, line)
                continue

            # Might be an intermediate: and %param, <mask>
            if src_reg in defs:
                and_match = re.match(r'and\s+\w+\s+(%\d+),\s*(\d+)', defs[src_reg])
                if and_match and pred == 'eq' and const_val == 0:
                    and_src, mask = and_match.group(1), int(and_match.group(2))
                    if and_src in params_by_reg:
                        param = params_by_reg[and_src]
                        param.constraints.append(ParamConstraint(
                            'multiple_of', mask + 1, raw_ir=f"  {defs[src_reg]}  ;;  {line}"
                        ))
                        continue

            # Check for ptrtoint -> and -> icmp eq 0  (alignment check)
            if src_reg in defs:
                and_match = re.match(r'and\s+\w+\s+(%\d+),\s*(\d+)', defs[src_reg])
                if and_match and pred == 'eq' and const_val == 0:
                    ptoi_reg = and_match.group(1)
                    mask = int(and_match.group(2))
                    if ptoi_reg in defs:
                        ptoi_match = re.match(r'ptrtoint\s+ptr\s+(%\d+)\s+to\s+i64', defs[ptoi_reg])
                        if ptoi_match and ptoi_match.group(1) in params_by_reg:
                            param = params_by_reg[ptoi_match.group(1)]
                            param.alignment = mask + 1
                            param.constraints.append(ParamConstraint(
                                'aligned', mask + 1, raw_ir=line
                            ))
                            continue

            continue

        # Pattern: icmp ne i32 %src, 0  (non-zero check, i.e. >= 1)
        ne_match = re.match(r'icmp\s+ne\s+\w+\s+(%\d+),\s*0', cond_def)
        if ne_match:
            src_reg = ne_match.group(1)
            if src_reg in params_by_reg:
                param = params_by_reg[src_reg]
                param.constraints.append(ParamConstraint('range_lo', 1, raw_ir=line))
                continue


def _add_icmp_constraint(param, pred, const_val, raw_ir):
    """Convert an icmp predicate + constant into a range constraint."""
    if pred in ('uge', 'sge'):
        param.constraints.append(ParamConstraint('range_lo', const_val, raw_ir=raw_ir))
    elif pred in ('ugt', 'sgt'):
        param.constraints.append(ParamConstraint('range_lo', const_val + 1, raw_ir=raw_ir))
    elif pred in ('ule', 'sle'):
        param.constraints.append(ParamConstraint('range_hi', const_val, raw_ir=raw_ir))
    elif pred in ('ult', 'slt'):
        param.constraints.append(ParamConstraint('range_hi', const_val - 1, raw_ir=raw_ir))
    elif pred == 'ne' and const_val == 0:
        param.constraints.append(ParamConstraint('range_lo', 1, raw_ir=raw_ir))


def analyze_thread_idx_ranges(lines):
    """
    Look for !range metadata on threadIdx/blockIdx reads.
    These use llvm.nvvm.read.ptx.sreg.* intrinsics.
    """
    thread_info = {}
    for line in lines:
        m = re.search(r'(%\d+)\s*=.*@llvm\.nvvm\.read\.ptx\.sreg\.(\w+)\(\)', line)
        if m:
            reg, sreg = m.group(1), m.group(2)
            # Check for !range metadata
            range_match = re.search(r'!range\s+!(\d+)', line)
            thread_info[sreg] = {
                'register': reg,
                'has_range_metadata': range_match is not None,
            }
    return thread_info


def analyze_branch_weights(lines):
    """Find branch weight metadata (!prof !N) for __builtin_expect."""
    branches = []
    for i, line in enumerate(lines):
        if '!prof' in line:
            branches.append((i + 1, line.strip()))
    return branches


def print_report(kernel_name, params, thread_info, branches, metadata_section):
    """Print a human-readable analysis report."""
    # Demangle kernel name
    demangled = kernel_name
    try:
        import subprocess
        result = subprocess.run(['c++filt', kernel_name], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            demangled = result.stdout.strip()
    except Exception:
        pass

    print("=" * 72)
    print(f"  Kernel Parameter Range Analysis")
    print(f"  Kernel: {demangled}")
    print(f"  Mangled: {kernel_name}")
    print("=" * 72)

    # Classify params
    ptr_params = [p for p in params if p.llvm_type == 'ptr']
    int_params = [p for p in params if p.llvm_type != 'ptr']

    # Pointer parameters
    if ptr_params:
        print(f"\n  Pointer parameters ({len(ptr_params)}):")
        print("  " + "-" * 60)
        for p in ptr_params:
            noalias = " [noalias/__restrict__]" if p.is_noalias else ""
            align = f" [align {p.alignment}B]" if p.alignment else ""
            print(f"    param {p.param_index} ({p.ir_reg}): {p.llvm_type}{noalias}{align}")
            if p.constraints:
                for c in p.constraints:
                    if c.kind == 'aligned':
                        print(f"      → aligned to {c.value} bytes")

    # Integer parameters (dimensions)
    if int_params:
        print(f"\n  Integer parameters — dimension ranges ({len(int_params)}):")
        print("  " + "-" * 60)
        for p in int_params:
            lo, hi = p.range_lo(), p.range_hi()
            mults = p.multiples()
            print(f"    param {p.param_index} ({p.ir_reg}): {p.llvm_type}")
            if lo is not None:
                print(f"      → min value: {lo}")
            if hi is not None:
                print(f"      → max value: {hi}")
            if mults:
                for m in mults:
                    print(f"      → must be multiple of {m}")
            if lo is not None and hi is not None:
                print(f"      ⇒ range [{lo}, {hi}]", end="")
                if mults:
                    print(f", stride {mults[0]}", end="")
                    possible_count = len(range(lo, hi + 1, mults[0]))
                    print(f"  ({possible_count} possible values)")
                else:
                    print()

    # Thread/block index info
    if thread_info:
        print(f"\n  Thread/Block index registers:")
        print("  " + "-" * 60)
        sreg_names = {
            'tid.x': 'threadIdx.x', 'tid.y': 'threadIdx.y', 'tid.z': 'threadIdx.z',
            'ctaid.x': 'blockIdx.x', 'ctaid.y': 'blockIdx.y', 'ctaid.z': 'blockIdx.z',
            'ntid.x': 'blockDim.x', 'ntid.y': 'blockDim.y', 'ntid.z': 'blockDim.z',
            'nctaid.x': 'gridDim.x', 'nctaid.y': 'gridDim.y', 'nctaid.z': 'gridDim.z',
        }
        for sreg, info in sorted(thread_info.items()):
            cuda_name = sreg_names.get(sreg.replace('_', '.'), sreg)
            range_note = " (has !range metadata)" if info['has_range_metadata'] else ""
            print(f"    {cuda_name}: {info['register']}{range_note}")

    # Branch weights
    if branches:
        print(f"\n  Branch weight hints (__builtin_expect):")
        print("  " + "-" * 60)
        for lineno, line in branches:
            print(f"    line {lineno}: {line}")

    # Named metadata
    if metadata_section:
        print(f"\n  LLVM Metadata (relevant entries):")
        print("  " + "-" * 60)
        for entry in metadata_section:
            print(f"    {entry}")

    print()
    print("=" * 72)
    print("  NOTE: llvm.assume() is an LLVM-IR-only construct.")
    print("  It does NOT appear in PTX.  Analysis must be done at IR level.")
    print("  PTX-visible annotations: .maxntid, .minnctapersm, .ptr .align")
    print("=" * 72)
    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <device.ll>", file=sys.stderr)
        sys.exit(1)

    ir_file = sys.argv[1]
    with open(ir_file) as f:
        lines = f.readlines()

    # Step 1: Parse kernel signatures
    kernels = parse_kernel_signature(lines)
    if not kernels:
        print("No kernel functions found in the IR.", file=sys.stderr)
        sys.exit(1)

    # Step 2: For each kernel, trace assume constraints
    for kernel_name, params in kernels.items():
        params_by_reg = {p.ir_reg: p for p in params}
        trace_assume_constraints(lines, params_by_reg)

        # Step 3: Thread index info
        thread_info = analyze_thread_idx_ranges(lines)

        # Step 4: Branch weights
        branches = analyze_branch_weights(lines)

        # Step 5: Relevant metadata
        metadata = []
        for line in lines:
            line = line.strip()
            if line.startswith('!') and any(kw in line for kw in [
                'branch_weights', 'range', 'nvvm.annotations', 'llvm.loop'
            ]):
                metadata.append(line)

        # Print report
        print_report(kernel_name, params, thread_info, branches, metadata)

    # Also print quick grep for key patterns
    print("Quick grep summary of annotation artifacts in IR:")
    print("-" * 50)
    counts = {
        'llvm.assume': 0,
        'noalias': 0,
        '!range': 0,
        '!prof': 0,
        'nsw': 0,   # no signed wrap
        'nuw': 0,   # no unsigned wrap
        'nonnull': 0,
    }
    for line in lines:
        for key in counts:
            if key in line:
                counts[key] += 1
    for key, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {key:20s}: {count:4d} occurrences")


if __name__ == '__main__':
    main()
