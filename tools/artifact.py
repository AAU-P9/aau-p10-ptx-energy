#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import sys
import time

def copy_build_folder(build_dir: Path, output_dir: Path):
    if not build_dir.exists() or not build_dir.is_dir():
        print(f"Build folder not found: {build_dir}", file=sys.stderr)
        return 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{build_dir.name}_bundle_{timestamp}"
    bundle_path = output_dir / bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(build_dir, bundle_path)
    print(f"Bundle created: {bundle_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Copy build folder contents into an artifact bundle.")
    parser.add_argument(
        "build_folder",
        help="Build folder to copy (e.g., build/vector_mult)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to place artifact bundles",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    return copy_build_folder(build_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
