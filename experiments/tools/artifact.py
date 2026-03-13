#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import sys
import time
from dataclasses import dataclass


@dataclass
class ArtifactConfig:
    """Configuration for the artifact bundler."""
    """build_folder: Path to the build directory to copy into an artifact bundle."""
    build_folder: Path
    """src_folder: Path to the source code directory to include in the artifact bundle."""
    src_folder: Path
    """output_dir: Path to the directory where artifact bundles will be placed."""
    output_dir: Path


def run_artifact(config: ArtifactConfig) -> int:
    """Copy build folder and source folder contents into an artifact bundle.
    
    Args:
        config: ArtifactConfig object containing all necessary parameters
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    build_dir = config.build_folder.resolve()
    if not build_dir.exists() or not build_dir.is_dir():
        print(f"Build folder not found: {build_dir}", file=sys.stderr)
        return 2
    
    src_dir = config.src_folder.resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        return 2

    output_dir = config.output_dir.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{build_dir.name}_bundle_{timestamp}"
    bundle_path = output_dir / bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.mkdir(parents=True, exist_ok=True)

    # Copy build folder contents
    shutil.copytree(build_dir, bundle_path / "build", dirs_exist_ok=True)
    # Copy source folder contents
    shutil.copytree(src_dir, bundle_path / "src", dirs_exist_ok=True)
    print(f"Bundle created: {bundle_path}")
    return 0


def parse_args() -> ArtifactConfig:
    """Parse command-line arguments and return configuration.
    
    Returns:
        ArtifactConfig object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Copy build folder and source folder contents into an artifact bundle.")
    parser.add_argument(
        "build_folder",
        help="Build folder to copy (e.g., experiments/build/vector_mult)",
    )
    parser.add_argument(
        "src_folder",
        help="Source folder to include (e.g., experiments/apps/vector_mult/src)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/artifacts",
        help="Directory to place artifact bundles",
    )
    args = parser.parse_args()

    return ArtifactConfig(
        build_folder=Path(args.build_folder),
        src_folder=Path(args.src_folder),
        output_dir=Path(args.output_dir),
    )


def main():
    """Main entry point."""
    config = parse_args()
    return run_artifact(config)


if __name__ == "__main__":
    raise SystemExit(main())
