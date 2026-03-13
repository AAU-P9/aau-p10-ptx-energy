# Compile, Run, and Package Tool

This repository includes a small Python utility that compiles a project folder, runs the resulting binary, captures its output, and produces a zip bundle containing the source code, binary, logs, and (for CUDA) PTX output.

## Usage

Stay at the repository root and run:

uv venv
uv pip install pandas plotly scipy

uv run experiments/tools/runner.py ./apps/<app_dir>

## Sambashare

You may often want to review your artifacts locally, for this using sambashare is a good option.
Simply add a system link to your build artifacts on your sambashare and it does the synchronization.

ln -s /home/p10/sambashare/artifacts experiments/artifacts
ln -s /home/p10/sambashare/plots experiments/plots

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

uv run experiments/tools/plotter.py ./build/<app_dir>

## Tasks

This repository includes "tasks" for VSCode allowing you to quickly run common tasks.

Available tasks:

1. **PTX: Run application** - Compiles and runs an application from the `experiments/apps/` directory. Prompts for the application name (e.g., `vector_add`).

2. **PTX: Create artifact** - Creates a bundled artifact (zip) from a built application in the `experiments/build/` directory. Prompts for the application name.

3. **PTX: Create plot** - Generates plots from an artifact bundle. Prompts for the artifact path (e.g., `experiments/artifacts/vector_add_bundle_...`).

4. **PTX: Create analysis** - Analyzes a PTX file's control flow graph with specified kernel parameters. Prompts for artifact path and kernel parameters (grid/block dimensions and parameters).

5. **PTX: Create Control Flow Graph** - Generates an HTML visualization of the control flow graph for a PTX file. Prompts for the artifact path and outputs `cfg.html` to the artifact directory.

To run a task, open the Command Palette (Ctrl+Shift+P) and select "Tasks: Run Task", then choose the desired task from the list.
