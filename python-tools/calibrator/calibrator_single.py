from pathlib import Path

from cubindings.cubindings import execute_code
from cubindings.cubindings_analyser import run_ptx_analyser, build_kernel_params, anayser_result_to_feature_csv
from calibrator.calibrator import build_program

csv_path = Path("/home/rasmus/aau-p10-ptx-energy/python-tools/nn-single-occurrences/data.csv")

def main():
    configs = [
        ("add.f32", 8, 1024),
        ("add.f32", 16, 1024),
        ("add.f32", 32, 1024),
        ("add.f32", 64, 1024),
        ("add.f32", 128, 1024),
        ("add.f32", 256, 1024),
        ("add.f32", 512, 1024),
        ("add.f32", 1024, 1024),
    ]
    for inst, grid, block in configs:
        print(f"Running {inst} with block={block} and grid={grid}")

        src = build_program(inst, iters=10_000_000, repeat=1, grid=grid, block=block)

        print("Executing kernel...")

        r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, debug=True)
        print(r.path)

        print(f"Kernel execution completed. Energy (J): {r.power_metric_result.total_energy_j:.3e}")

        print("Running PTX analyser...")

        p = run_ptx_analyser(
            r.path,
            kernel_params=build_kernel_params(r.exports),
        )

        print(f"PTX analyser completed.")

        anayser_result_to_feature_csv(f"{inst}_b{block}_g{grid}", csv_path, r.power_metric_result.total_energy_j, p)


if __name__ == "__main__":
    results = main()
