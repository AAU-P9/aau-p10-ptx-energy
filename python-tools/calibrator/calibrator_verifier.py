from pathlib import Path

from calibrator.calibrator import build_program, INSTRUCTION_TEMPLATES
from cubindings.cubindings_analyser import run_ptx_analyser, build_kernel_params
from cubindings.cubindings_predictor import run_predictor
from cubindings.cubindings import execute_code

MODEL_PATH=Path("/home/rasmus/aau-p10-ptx-energy/linear-model/linear-model.py")
WEIGHTS_PATH=Path("/home/p10/aau-p10-ptx-energy/linear-model/weights.csv")
GRID = 152          # 2 blocks per SM on AD103 (76 SMs)
BLOCK = 1024

debug_enabled = False

for insn in INSTRUCTION_TEMPLATES:
    print(f"\n=== {insn} ===")

    src = build_program(
        insn=insn,
        block=BLOCK,
        grid=GRID,
        iters=23566108,
    )

    r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True)

    ar = run_ptx_analyser(
        r.path,
        kernel_params=build_kernel_params(r.exports),
        debug_enabled=debug_enabled,
    )

    pr = run_predictor(
        model_path=MODEL_PATH,
        output_path=r.path,
        weights_path=WEIGHTS_PATH,
        debug_enabled=debug_enabled,
    )

    print("Actual total power (J):", r.power_metric_result.total_energy_j)
    print("Predicted total power (J):", pr.total_estimated_power_joules)
    print("Predicted power breakdown:")
    for estimate in pr.estimates:
        print(f"  {estimate.instruction}: {estimate.estimated_power_joules:.0f} J (count: {estimate.count:.0f}, avg per occurrence: {estimate.avg_power_per_occurrence_joules:.3e} J, used fallback: {estimate.used_fallback})")