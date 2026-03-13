#!/usr/bin/env python3
"""Extract power metrics from GPU power data from nvidia-smi and pmd2 CSVs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from scipy import integrate


@dataclass
class PowerMetrics:
    """Power and energy metrics from kernel execution."""
    kernel_start_gpu_ns: float
    kernel_end_gpu_ns: float
    kernel_duration_gpu_ns: float
    kernel_start_cpu_s: float
    kernel_end_cpu_s: float
    kernel_duration_cpu_s: float
    duration_error_s: float
    total_energy_j: float
    sensor4_energy_j: float
    sensor7_energy_j: float
    num_samples: int
    regression_slope: float
    regression_intercept: float
    regression_samples: int


@dataclass
class PowerConfig:
    """Configuration for the power metrics extractor."""
    path: Path


@dataclass
class IncrementalRegression:
    """Incremental linear regression for timestamp conversion."""
    n: int = 0
    x_mean: float = 0.0
    y_mean: float = 0.0
    x_svar: float = 0.0
    y_svar: float = 0.0
    xy_scov: float = 0.0

    def add_sample(self, x: float, y: float) -> None:
        """Add a sample to the regression."""
        self.n += 1
        x_mean_prev = self.x_mean
        y_mean_prev = self.y_mean
        self.x_mean += (x - self.x_mean) / self.n
        self.y_mean += (y - self.y_mean) / self.n
        self.x_svar += (x - x_mean_prev) * (x - self.x_mean)
        self.y_svar += (y - y_mean_prev) * (y - self.y_mean)
        self.xy_scov += (x - x_mean_prev) * (y - self.y_mean)

    def parameters(self) -> tuple[float, float]:
        """Get linear regression parameters (slope, intercept)."""
        if self.x_svar == 0:
            return 0.0, self.y_mean
        slope = self.xy_scov / self.x_svar
        intercept = self.y_mean - slope * self.x_mean
        return slope, intercept

    def orthogonal(self) -> tuple[float, float]:
        """Get orthogonal (Deming) regression parameters with delta=1."""
        import math
        delta = 1.0
        k = self.y_svar - delta * self.x_svar
        slope = (k + math.sqrt(k * k + 4 * delta * self.xy_scov * self.xy_scov)) / (2 * self.xy_scov)
        intercept = self.y_mean - slope * self.x_mean
        return slope, intercept


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _get_column(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of the columns found: {', '.join(candidates)}")


def load_nvidia_smi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df["timestamp"] = pd.to_datetime(_get_column(df, "timestamp"), errors="coerce")
    df["power_draw_w"] = _to_float(_get_column(df, "power.draw [W]", "power.draw[W]", "power.draw"))
    df["temperature_c"] = pd.to_numeric(_get_column(df, "temperature.gpu", "temperature"), errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    t0 = df["timestamp"].iloc[0]
    df["t_s"] = (df["timestamp"] - t0).dt.total_seconds()
    return df


def load_pmd2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df = df.dropna(subset=["timestamp_ns"]).copy()
    t0 = df["timestamp_ns"].iloc[0]
    df["t_s"] = (df["timestamp_ns"] - t0) / 1_000_000_000.0
    df["total_power_w"] = pd.to_numeric(df["total_power_w"], errors="coerce")
    df["sensor4_power_w"] = pd.to_numeric(df["sensor4_power_mw"], errors="coerce") / 1000.0
    df["sensor7_power_w"] = pd.to_numeric(df["sensor7_power_mw"], errors="coerce") / 1000.0
    return df


def parse_output_txt(path: Path) -> dict:
    """Parse output.txt for OFFSET and KERNEL timing information."""
    result = {
        "offsets": [],
        "kernel_start": None,
        "kernel_end": None,
        "kernel_duration": None,
    }
    
    if not path.exists():
        return result
    
    with open(path, "r") as f:
        for line in f:
            # Parse OFFSET lines
            offset_match = re.match(
                r"\[OFFSET\] CUPTI Timestamp: (\d+), CPU Timestamp #1: (\d+), CPU Timestamp #2: (\d+)",
                line,
            )
            if offset_match:
                cupti_ts = int(offset_match.group(1))
                cpu_ts1 = int(offset_match.group(2))
                cpu_ts2 = int(offset_match.group(3))
                # Use average of CPU timestamps
                cpu_ts_avg = (cpu_ts1 * 3.0 + cpu_ts2 * 5.0) / 8.0
                result["offsets"].append((cupti_ts, cpu_ts_avg))
            
            # Parse KERNEL lines
            kernel_start = re.match(r"\[KERNEL\] Start Time: (\d+)", line)
            if kernel_start:
                result["kernel_start"] = int(kernel_start.group(1))
            
            kernel_end = re.match(r"\[KERNEL\] End Time: (\d+)", line)
            if kernel_end:
                result["kernel_end"] = int(kernel_end.group(1))
            
            kernel_duration = re.match(r"\[KERNEL\] Duration: (\d+)", line)
            if kernel_duration:
                result["kernel_duration"] = int(kernel_duration.group(1))
    
    return result


def gpu_to_cpu_time(gpu_timestamp: float, slope: float, intercept: float) -> float:
    """Convert GPU timestamp to CPU timestamp using linear regression."""
    return slope * gpu_timestamp + intercept


def extract_power_metrics(config: PowerConfig) -> PowerMetrics:
    """Extract power metrics from GPU data.

    Args:
        config: PowerConfig object containing path to data directory

    Returns:
        PowerMetrics object containing calculated metrics
    """
    nv_path = config.path / "nvidia-smi.csv"
    pmd2_path = config.path / "pmd2.csv"
    output_txt_path = config.path / "output.txt"
    
    nv = load_nvidia_smi(nv_path)
    pmd2 = load_pmd2(pmd2_path)
    
    # Parse timing information
    output_info = parse_output_txt(output_txt_path)

    # Build incremental regression model
    regression = IncrementalRegression()
    for gpu_ts, cpu_ts in output_info["offsets"]:
        regression.add_sample(float(gpu_ts), float(cpu_ts))

    # Get regression parameters (using orthogonal regression for better accuracy)
    slope, intercept = regression.orthogonal()

    # Convert kernel GPU timestamps to CPU timestamps
    kernel_start_gpu = float(output_info["kernel_start"])
    kernel_end_gpu = float(output_info["kernel_end"])
    kernel_start_cpu_ns = gpu_to_cpu_time(kernel_start_gpu, slope, intercept)
    kernel_end_cpu_ns = gpu_to_cpu_time(kernel_end_gpu, slope, intercept)

    # pmd2's first timestamp in nanoseconds
    pmd2_t0_ns = pmd2["timestamp_ns"].iloc[0]

    # Normalize to pmd2's time axis (relative time since first sample)
    kernel_start_pmd2_ns = kernel_start_cpu_ns - pmd2_t0_ns
    kernel_end_pmd2_ns = kernel_end_cpu_ns - pmd2_t0_ns

    # Convert to seconds for processing
    kernel_start_cpu_s = (kernel_start_cpu_ns - pmd2_t0_ns) / 1e9
    kernel_end_cpu_s = (kernel_end_cpu_ns - pmd2_t0_ns) / 1e9
    kernel_duration_cpu_s = kernel_end_cpu_s - kernel_start_cpu_s

    # Calculate energy consumed during kernel execution using trapezoidal integration
    kernel_mask = (pmd2["t_s"] >= kernel_start_cpu_s) & (pmd2["t_s"] <= kernel_end_cpu_s)
    pmd2_kernel = pmd2[kernel_mask].copy()
    
    total_energy_j = 0.0
    sensor4_energy_j = 0.0
    sensor7_energy_j = 0.0
    num_samples = len(pmd2_kernel)
    
    if len(pmd2_kernel) > 1:
        # Remove the last line in pmd2 since it is often corrupt
        pmd2_kernel = pmd2_kernel.iloc[:-1]
        num_samples = len(pmd2_kernel)

        # Integrate power over time to get energy (Joules = Watts * seconds)
        total_energy_j = integrate.trapezoid(pmd2_kernel["total_power_w"], pmd2_kernel["t_s"])
        sensor4_energy_j = integrate.trapezoid(pmd2_kernel["sensor4_power_w"], pmd2_kernel["t_s"])
        sensor7_energy_j = integrate.trapezoid(pmd2_kernel["sensor7_power_w"], pmd2_kernel["t_s"])

    # Calculate duration error
    duration_error_s = abs(kernel_duration_cpu_s - output_info["kernel_duration"] / 1e9)

    return PowerMetrics(
        kernel_start_gpu_ns=kernel_start_gpu,
        kernel_end_gpu_ns=kernel_end_gpu,
        kernel_duration_gpu_ns=float(output_info["kernel_duration"]),
        kernel_start_cpu_s=kernel_start_cpu_s,
        kernel_end_cpu_s=kernel_end_cpu_s,
        kernel_duration_cpu_s=kernel_duration_cpu_s,
        duration_error_s=duration_error_s,
        total_energy_j=total_energy_j,
        sensor4_energy_j=sensor4_energy_j,
        sensor7_energy_j=sensor7_energy_j,
        num_samples=num_samples,
        regression_slope=slope,
        regression_intercept=intercept,
        regression_samples=len(output_info["offsets"]),
    )


def parse_args() -> PowerConfig:
    """Parse command-line arguments and return configuration.

    Returns:
        PowerConfig object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract power metrics from GPU data")
    parser.add_argument("path", type=Path, help="Path to the directory containing CSV files")
    args = parser.parse_args()

    return PowerConfig(
        path=args.path,
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()
    metrics = extract_power_metrics(config)
    
    # Print metrics in a readable format
    print(f"Kernel Timing (GPU):")
    print(f"  Start: {metrics.kernel_start_gpu_ns / 1e9:.9f} s")
    print(f"  End: {metrics.kernel_end_gpu_ns / 1e9:.9f} s")
    print(f"  Duration: {metrics.kernel_duration_gpu_ns / 1e9:.9f} s")
    
    print(f"\nKernel Timing (Estimated CPU):")
    print(f"  Start: {metrics.kernel_start_cpu_s:.9f} s")
    print(f"  End: {metrics.kernel_end_cpu_s:.9f} s")
    print(f"  Duration: {metrics.kernel_duration_cpu_s:.9f} s")
    print(f"  Duration error: {metrics.duration_error_s:.9f} s")
    
    print(f"\nRegression Model:")
    print(f"  Slope: {metrics.regression_slope:.10f}")
    print(f"  Intercept: {metrics.regression_intercept:.2f}")
    print(f"  Samples used: {metrics.regression_samples}")
    
    print(f"\nEnergy Consumption:")
    print(f"  Total energy: {metrics.total_energy_j:.6f} J")
    print(f"  Sensor 4 energy: {metrics.sensor4_energy_j:.6f} J")
    print(f"  Sensor 7 energy: {metrics.sensor7_energy_j:.6f} J")
    print(f"  Samples used for integration: {metrics.num_samples}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
