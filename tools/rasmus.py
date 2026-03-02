#!/usr/bin/env python3
"""End-to-end pipeline: compile & run, bundle artifacts, and plot results."""

import argparse
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import runner
import artifact
import plotter


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    """folder: Path to the folder containing source code to compile and run."""
    folder: Path
    """output_dir: Path to the directory where output artifacts will be placed."""
    output_dir: Path
    """run_args: Optional list of arguments to pass to the compiled binary."""
    run_args: Optional[List[str]] = None


def parse_args() -> PipelineConfig:
    """Parse command-line arguments and return configuration.

    Returns:
        PipelineConfig object containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: compile & run an app, bundle build artifacts, and plot results.",
    )
    parser.add_argument(
        "folder",
        help="Folder containing source code to compile and run (e.g., apps/vector_add)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to place artifact bundles (default: artifacts)",
    )
    parser.add_argument(
        "--run-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the compiled binary",
    )
    args = parser.parse_args()

    return PipelineConfig(
        folder=Path(args.folder),
        output_dir=Path(args.output_dir),
        run_args=args.run_args,
    )


def run_pipeline(config: PipelineConfig) -> int:
    """Run the full pipeline: compile & run, bundle artifacts, plot results.

    Args:
        config: PipelineConfig object containing all necessary parameters

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Step 1: Compile and run
    run_config = runner.RunnerConfig(
        folder=config.folder,
        output_dir=config.output_dir,
        run_args=config.run_args,
    )
    rc = runner.run_runner(run_config)
    if rc != 0:
        return rc

    # Step 2: Bundle build artifacts
    # The runner places build output in build/<folder_name>/
    build_dir = Path("build") / config.folder.resolve().name
    artifact_config = artifact.ArtifactConfig(
        build_folder=build_dir,
        output_dir=config.output_dir,
    )
    rc = artifact.run_artifact(artifact_config)
    if rc != 0:
        return rc

    # Step 3: Plot results from the build directory
    plot_config = plotter.PlotterConfig(
        path=build_dir,
    )
    rc = plotter.run_plotter(plot_config)
    if rc != 0:
        return rc

    return 0


def main() -> int:
    """Main entry point."""
    config = parse_args()
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())