#!/usr/bin/env python3
"""Plot combined GPU power data from nvidia-smi and pmd2 CSVs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from scipy import integrate


NVIDIA_DEFAULT = Path("build/example/nvidia-smi.csv")
PMD2_DEFAULT = Path("build/example/pmd2.csv")


@dataclass
class PlotterConfig:
    """Configuration for the plotter."""
    """path: Path to the directory containing CSV files (nvidia-smi.csv, pmd2.csv, output.txt)."""
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


def build_figure(nv: pd.DataFrame, pmd2: pd.DataFrame, timing_data: dict | None = None, t0_cpu_ns: float = 0) -> go.Figure:
    fig = go.Figure()

    if not nv.empty:
        fig.add_trace(
            go.Scatter(
                x=nv["t_s"],
                y=nv["power_draw_w"],
                mode="lines",
                name="nvidia-smi power.draw [W]",
            )
        )

    if not pmd2.empty:
        fig.add_trace(
            go.Scatter(
                x=pmd2["t_s"],
                y=pmd2["total_power_w"],
                mode="lines",
                name="pmd2 total_power [W]",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pmd2["t_s"],
                y=pmd2["sensor4_power_w"],
                mode="lines",
                name="pmd2 sensor4_power [W]",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pmd2["t_s"],
                y=pmd2["sensor7_power_w"],
                mode="lines",
                name="pmd2 sensor7_power [W]",
            )
        )

    # Add kernel start/end lines if timing data is available
    if timing_data and timing_data.get("kernel_start_cpu_s") is not None:
        kernel_start_s = timing_data["kernel_start_cpu_s"]
        kernel_end_s = timing_data["kernel_end_cpu_s"]
        
        fig.add_vline(
            x=kernel_start_s,
            line_dash="dash",
            line_color="green",
            annotation_text="Kernel Start",
            annotation_position="top",
        )
        fig.add_vline(
            x=kernel_end_s,
            line_dash="dash",
            line_color="red",
            annotation_text="Kernel End",
            annotation_position="top",
        )

    fig.update_layout(
        title="Combined GPU Power Metrics",
        xaxis_title="Time since start [s]",
        yaxis_title="Power [W]",
        yaxis2=dict(
            title="Temperature [C]",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig


def run_plotter(config: PlotterConfig) -> int:
    """Plot combined GPU power data from nvidia-smi and pmd2 CSVs.

    Args:
        config: PlotterConfig object containing all necessary parameters

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    output_dir = Path(f"plots/{config.path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_file = output_dir / "analysis.log"
    
    def log_print(message: str) -> None:
        """Print to both console and log file."""
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")
    
    nv_path = config.path / "nvidia-smi.csv"
    pmd2_path = config.path / "pmd2.csv"
    output_txt_path = config.path / "output.txt"
    
    nv = load_nvidia_smi(nv_path)
    pmd2 = load_pmd2(pmd2_path)
    
    # Parse timing information and calibrate GPU->CPU timestamp conversion
    output_info = parse_output_txt(output_txt_path)
    timing_data = None

    # Print the GPU timestamps for debugging (Note that the start and end timestamp are not normalized, but the duration should be fine.)
    log_print(f"(CUPTI GPU) Kernel start timestamp: {output_info['kernel_start'] / 1_000_000_000}")
    log_print(f"(CUPTI GPU) Kernel end timestamp: {output_info['kernel_end'] / 1_000_000_000}")
    log_print(f"(CUPTI GPU) Kernel duration: {output_info['kernel_duration'] / 1_000_000_000}")

    # Build incremental regression model
    regression = IncrementalRegression()
    for gpu_ts, cpu_ts in output_info["offsets"]:
        regression.add_sample(float(gpu_ts), float(cpu_ts))

    # Get regression parameters (using orthogonal regression for better accuracy)
    slope, intercept = regression.orthogonal()

    log_print(f"\n[DEBUG] Regression parameters:")
    log_print(f"  Slope: {slope:.10f}")
    log_print(f"  Intercept: {intercept:.2f}")
    log_print(f"  Samples used: {len(output_info['offsets'])}")

    # Convert kernel GPU timestamps to CPU timestamps
    kernel_start_gpu = float(output_info["kernel_start"])
    kernel_end_gpu = float(output_info["kernel_end"])
    kernel_start_cpu_ns = gpu_to_cpu_time(kernel_start_gpu, slope, intercept)
    kernel_end_cpu_ns = gpu_to_cpu_time(kernel_end_gpu, slope, intercept)

    # pmd2's first timestamp in microseconds
    pmd2_t0_ns = pmd2["timestamp_ns"].iloc[0]

    # Normalize to pmd2's time axis (relative time since first sample)
    kernel_start_pmd2_ns = kernel_start_cpu_ns - pmd2_t0_ns
    kernel_end_pmd2_ns = kernel_end_cpu_ns - pmd2_t0_ns

    # Convert to seconds for plotting
    kernel_start_cpu_s = (kernel_start_cpu_ns - pmd2_t0_ns) / 1e9
    kernel_end_cpu_s = (kernel_end_cpu_ns - pmd2_t0_ns) / 1e9

    timing_data = {
        "kernel_start_cpu_s": kernel_start_cpu_s,
        "kernel_end_cpu_s": kernel_end_cpu_s,
        "kernel_duration_s": kernel_end_cpu_s - kernel_start_cpu_s,
    }

    log_print(f"\n(Estimated CPU) Kernel start: {timing_data['kernel_start_cpu_s']:.6f} s")
    log_print(f"(Estimated CPU) Kernel end: {timing_data['kernel_end_cpu_s']:.6f} s")
    log_print(f"(Estimated CPU) Kernel duration: {timing_data['kernel_duration_s']:.6f} s")

    # Print the duration error
    duration_error_s = abs(timing_data["kernel_duration_s"] - output_info["kernel_duration"] / 1e9)
    log_print(f"Duration error: {duration_error_s:.6f} s")

    # Calculate energy consumed during kernel execution using trapezoidal integration
    log_print(f"\n[Energy Consumption]")
    kernel_mask = (pmd2["t_s"] >= kernel_start_cpu_s) & (pmd2["t_s"] <= kernel_end_cpu_s)
    pmd2_kernel = pmd2[kernel_mask].copy()
    
    if len(pmd2_kernel) > 1:
        # Remove the last line in pmd2 since it is often corrupt
        pmd2_kernel = pmd2_kernel.iloc[:-1]

        # Integrate power over time to get energy (Joules = Watts * seconds)
        energy_total_j = integrate.trapezoid(pmd2_kernel["total_power_w"], pmd2_kernel["t_s"])
        energy_sensor4_j = integrate.trapezoid(pmd2_kernel["sensor4_power_w"], pmd2_kernel["t_s"])
        energy_sensor7_j = integrate.trapezoid(pmd2_kernel["sensor7_power_w"], pmd2_kernel["t_s"])
        
        log_print(f"Total energy (pmd2 total_power): {energy_total_j:.6f} J")
        log_print(f"Sensor 4 energy: {energy_sensor4_j:.6f} J")
        log_print(f"Sensor 7 energy: {energy_sensor7_j:.6f} J")
        log_print(f"Samples used for integration: {len(pmd2_kernel)}")
    else:
        log_print(f"Not enough samples in kernel window for integration (found {len(pmd2_kernel)} samples)")
    
    # Create separate figures for power and temperature
    power_fig = build_figure(nv, pmd2, timing_data)
    power_fig.update_layout(title="Power Metrics")
    power_fig.write_html(output_dir / "power_metrics.html", include_plotlyjs="cdn")
    
    temp_fig = go.Figure()
    temp_fig.add_trace(
        go.Scatter(
            x=nv["t_s"],
            y=nv["temperature_c"],
            mode="lines",
            name="nvidia-smi temperature [C]",
        )
    )
    temp_fig.update_layout(title="Temperature Metrics", xaxis_title="Time since start [s]", yaxis_title="Temperature [C]")
    temp_fig.write_html(output_dir / "temperature_metrics.html", include_plotlyjs="cdn")
    log_print(f"Wrote plots to {output_dir}")
    log_print(f"Log file written to {log_file}")
    return 0


def parse_args() -> PlotterConfig:
    """Parse command-line arguments and return configuration.

    Returns:
        PlotterConfig object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Plot combined CSV data with Plotly")
    parser.add_argument("path", type=Path, help="Path to the directory containing CSV files")
    args = parser.parse_args()

    return PlotterConfig(
        path=args.path,
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()
    return run_plotter(config)


if __name__ == "__main__":
    raise SystemExit(main())
