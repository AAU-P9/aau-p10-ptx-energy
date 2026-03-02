#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path
import sys
import time

def zip_build_folder(build_dir: Path, output_dir: Path):
    if not build_dir.exists() or not build_dir.is_dir():
        print(f"Build folder not found: {build_dir}", file=sys.stderr)
        return 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{build_dir.name}_bundle_{timestamp}.zip"
    bundle_path = output_dir / bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in build_dir.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(build_dir.parent)
                zipf.write(path, arcname)
    print(f"Bundle created: {bundle_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Zip build folder contents into an artifact bundle.")
    parser.add_argument(
        "build_folder",
        help="Build folder to zip (e.g., build/vector_mult)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to place zip bundles",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    return zip_build_folder(build_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
