#!/usr/bin/env python3
"""Simple generator to create scaled copies of benchmark JSON files.

Usage:
  python generate_artificial.py /path/to/dir 2

Creates files named <orig>_x<multiplier>.json with scaled fields.
Multiples: powerConsumptionJoules, kernel duration keys, totalInstructions,
instructionOccurrences, dependentPairs, and independentPairs.
Grid and block dimensions remain unchanged.
"""
from pathlib import Path
import argparse
import json
import sys

# Hard-coded keys to match example JSON structure

# Energy key from example: "powerConsumptionJoules"
ENERGY_KEY = "powerConsumptionJoules"

# Duration key from example: "kernelDurationS"
DURATION_KEY = "kernelDurationS"

# Keys to multiply to represent scaled workloads
SCALE_KEYS = [
    "totalInstructions",
    "instructionOccurrences",
    "dependentPairs",
    "independentPairs",
]


def scale_file(path: Path, multiplier: int, out_dir: Path | None = None) -> Path:
    with path.open("r") as f:
        data = json.load(f)

    # Scale the energy key if present
    if ENERGY_KEY in data and isinstance(data[ENERGY_KEY], (int, float)):
        data[ENERGY_KEY] = data[ENERGY_KEY] * multiplier

    # Scale the duration key if present
    if DURATION_KEY in data and isinstance(data[DURATION_KEY], (int, float)):
        data[DURATION_KEY] = data[DURATION_KEY] * multiplier

    # Scale workload-related keys
    for key in SCALE_KEYS:
        if key not in data:
            continue
        
        if isinstance(data[key], int):
            # Simple integer value
            data[key] = data[key] * multiplier
        elif isinstance(data[key], dict):
            # Dictionary with counts as values
            for sub_key, value in data[key].items():
                if isinstance(value, (int, float)):
                    data[key][sub_key] = value * multiplier

    # Write scaled copy into out_dir if provided, otherwise same directory
    new_name = f"{path.stem}_x{multiplier}{path.suffix}"
    if out_dir is None:
        out_path = path.with_name(new_name)
    else:
        out_path = out_dir / new_name
        out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate scaled benchmark JSON copies")
    parser.add_argument("dir", help="Directory with JSON benchmark files")
    parser.add_argument("multiplier", type=int, help="Multiplier to scale durations/energy/grid.x by (integer)")
    parser.add_argument("--out", "-o", dest="out_dir", help="Output directory for generated files (optional)")
    args = parser.parse_args(argv)

    d = Path(args.dir)
    if not d.exists() or not d.is_dir():
        print(f"Error: {d} is not a directory", file=sys.stderr)
        return 2

    if args.multiplier < 1:
        print("Error: multiplier must be >= 1", file=sys.stderr)
        return 2

    files = sorted(d.glob("*.json"))
    if not files:
        print(f"No JSON files found in {d}")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        out = scale_file(p, args.multiplier, out_dir=out_dir)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
