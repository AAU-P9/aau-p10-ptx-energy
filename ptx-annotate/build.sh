#!/usr/bin/env bash
# build.sh — compile annotated_matmul.cu, generate PTX, optionally dump LLVM IR
#
# Primary compiler: nvcc  (only dependency: CUDA toolkit)
# Optional:         clang + LLVM tools  (for LLVM IR analysis)
#
# Usage:
#   chmod +x build.sh
#   ./build.sh                       # build + run with nvcc only
#   CUDA_PATH=/opt/cuda ./build.sh   # custom CUDA install
#   SM=89 ./build.sh                 # different GPU arch (default sm_86)
#   SRC=reduce.cu ./build.sh         # build a different kernel file
#   ANALYSIS=1 ./build.sh            # also generate LLVM IR with clang

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration — override via environment variables
# --------------------------------------------------------------------------
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
SM="${SM:-86}"
SRC="${SRC:-reduce.cu}"
OUT="build"
ANALYSIS="${ANALYSIS:-0}"    # set to 1 to also generate LLVM IR with clang

# Project include directory (for ptx_meta.h, cupti_timing.h, etc.)
PROJECT_INCLUDE="-I../include"

# Derive executable name from source file
EXE_NAME="${SRC%.cu}_demo"

ARCH="sm_${SM}"

mkdir -p "$OUT"

echo "========================================================"
echo "  Annotated CUDA build  |  arch=${ARCH}  |  src=${SRC}"
echo "  Compiler: nvcc   |   LLVM analysis: $([ "$ANALYSIS" = 1 ] && echo ON || echo 'OFF (set ANALYSIS=1 to enable)')"
echo "========================================================"

# ==========================================================================
# STEP 1 — PTX generation with nvcc
# ==========================================================================
echo ""
echo "[1/3] Generating PTX (nvcc)       ->  $OUT/device.ptx"
nvcc -arch="$ARCH" -std=c++17 -O2 \
    $PROJECT_INCLUDE \
    --ptx \
    "$SRC" -o "$OUT/device.ptx"

# Strip nvcc's "begin/end inline asm" wrapper comments for cleaner PTX
sed -i '/^[[:space:]]*\/\/ begin inline asm$/d; /^[[:space:]]*\/\/ end inline asm$/d' "$OUT/device.ptx"

echo ""
echo "  __launch_bounds__ in PTX:"
echo "  -------------------------------------------------------"
grep -E "\.maxntid|\.minnctapersm|\.maxnreg|\.visible .entry" "$OUT/device.ptx" || \
    echo "  (none found — add __launch_bounds__ to your kernel!)"

echo ""
echo "  PTX parameter annotations:"
echo "  -------------------------------------------------------"
grep -E "\.param|\.align|\.ptr" "$OUT/device.ptx" | head -20

echo ""
echo "  @META comments in PTX (for your parser):"
echo "  -------------------------------------------------------"
grep "@META" "$OUT/device.ptx" || echo "  (none found)"

echo ""
echo "  Constant metadata symbols:"
echo "  -------------------------------------------------------"
grep -E "__meta_|__ptxmeta_" "$OUT/device.ptx" || echo "  (none found)"

echo ""
echo "  Read-only loads (from __restrict__):"
echo "  -------------------------------------------------------"
count=$(grep -c "ld\.global\.nc" "$OUT/device.ptx" 2>/dev/null || true)
echo "  ld.global.nc (non-coherent): ${count:-0} occurrences"

# ==========================================================================
# STEP 2 — Compile executable with nvcc and run
# ==========================================================================
echo ""
echo "[2/3] Compiling executable (nvcc) ->  $OUT/$EXE_NAME"
nvcc -arch="$ARCH" -std=c++17 -O2 \
    $PROJECT_INCLUDE \
    "$SRC" -o "$OUT/$EXE_NAME"

echo "      Running $EXE_NAME..."
"$OUT/$EXE_NAME"
RET=$?
if [ $RET -eq 0 ]; then
    echo "      $EXE_NAME exited successfully (code $RET)"
else
    echo "      $EXE_NAME FAILED (exit code $RET)"
fi

# ==========================================================================
# STEP 3 — (optional) LLVM IR analysis with clang
#
# The DEVICE_ASSUME macros use `if (!(expr)) __builtin_unreachable()` which:
#   - nvcc:  uses as optimisation hint (no IR output available)
#   - clang: folds into llvm.assume(i1 %cond) in the LLVM IR
#
# If you need to analyse parameter ranges at the IR level, enable this
# step with ANALYSIS=1.  It requires: clang++, llvm-as, opt, llvm-dis
# ==========================================================================
if [ "$ANALYSIS" = "1" ]; then
    echo ""
    echo "========================================================"
    echo "  LLVM IR Analysis (clang)"
    echo "========================================================"

    CXX="${CXX:-clang++}"
    OPT="${OPT:-opt}"
    DIS="${DIS:-llvm-dis}"

    CLANG_FLAGS=(
        -x cuda
        --cuda-path="$CUDA_PATH"
        --cuda-gpu-arch="$ARCH"
        -I"$CUDA_PATH/include"
        $PROJECT_INCLUDE
        -std=c++17
        -O2
        -Wno-unknown-cuda-version
    )

    echo ""
    echo "[3a] Device LLVM IR              ->  $OUT/device.ll"
    "$CXX" "${CLANG_FLAGS[@]}" \
        --cuda-device-only \
        -emit-llvm -S \
        "$SRC" -o "$OUT/device.ll"

    echo "[3b] Device bitcode              ->  $OUT/device.bc"
    llvm-as "$OUT/device.ll" -o "$OUT/device.bc"

    echo "[3c] Optimised IR (O2)           ->  $OUT/device_opt.ll"
    "$OPT" --passes="default<O2>" \
        "$OUT/device.bc" -o "$OUT/device_opt.bc" 2>/dev/null || \
        "$OPT" -O2 "$OUT/device.bc" -o "$OUT/device_opt.bc"
    "$DIS" "$OUT/device_opt.bc" -o "$OUT/device_opt.ll"

    echo "[3d] Host LLVM IR                ->  $OUT/host.ll"
    "$CXX" "${CLANG_FLAGS[@]}" \
        --cuda-host-only \
        -emit-llvm -S \
        "$SRC" -o "$OUT/host.ll"

    echo ""
    echo "  llvm.assume / annotation count in device IR:"
    echo "  -------------------------------------------------------"
    for pat in "llvm.assume" "noalias" "!range" "!prof" "nsw" "nuw"; do
        count=$(grep -c "$pat" "$OUT/device.ll" 2>/dev/null || true)
        count="${count:-0}"
        printf "    %-30s %3s occurrences\n" "$pat" "$count"
    done

    echo ""
    echo "  Parameter Range Analysis:"
    echo "  -------------------------------------------------------"
    if command -v python3 &>/dev/null && [ -f analyze_ir.py ]; then
        python3 analyze_ir.py "$OUT/device.ll"
    else
        echo "  (python3 or analyze_ir.py not found)"
    fi
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "========================================================"
echo "  Output files:"
echo "========================================================"
echo "  $OUT/device.ptx      <- PTX assembly (nvcc)"
echo "  $OUT/$EXE_NAME     <- executable"
if [ "$ANALYSIS" = "1" ]; then
echo "  $OUT/device.ll       <- device LLVM IR  (clang, start here for analysis)"
echo "  $OUT/device.bc       <- device bitcode  (feed to opt / your pass)"
echo "  $OUT/device_opt.ll   <- IR after O2 opt passes"
echo "  $OUT/host.ll         <- host IR with cudaLaunchKernel + dim3 sizes"
fi
echo ""
echo "Next steps:"
echo "  # Re-run with LLVM IR analysis:"
echo "  ANALYSIS=1 ./build.sh"
echo ""
echo "  # Run your own opt pass on the IR:"
echo "  opt --load-pass-plugin=./my_pass.so --passes=my-range-analysis \\"
echo "      $OUT/device.bc -o /dev/null"
