# Compile, Run, and Package Tool

This repository includes a small Python utility that compiles a project folder, runs the resulting binary, captures its output, and produces a zip bundle containing the source code, binary, logs, and (for CUDA) PTX output.

## Usage

Stay at the repository root and run:

uv venv
uv pip install pandas plotly scipy

uv run tools/runner.py ./apps/<app_dir>

## Sambashare

You may often want to review your artifacts locally, for this using sambashare is a good option.
Simply add a system link to your build artifacts on your sambashare and it does the synchronization.

ln -s /home/p10/sambashare/artifacts artifacts
ln -s /home/p10/sambashare/plots plots

### Optional flags

- `--output-dir <path>`: where to write the build artifacts and bundle (default: `artifacts`).
- `--run-args ...`: arguments passed to the compiled binary.

## Outputs

The tool creates a folder under the output directory named after the target folder (e.g., `artifacts/app`) and a timestamped zip bundle in the output directory. The bundle includes:

- Source code from the target folder
- Compiled binary
- `compile.log`
- `output.txt`
- `manifest.json`
- `ptx.log` and `.ptx` files (when using `nvcc`)

## Graphs

Their is an additional tool to draw a graph of the outputs. This tool can be... TODO finish this

uv run tools/plotter.py ./build/<app_dir>
