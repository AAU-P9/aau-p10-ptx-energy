from pathlib import Path
from dataclasses import dataclass
from typing import Any
import pandas as pd
from scipy import integrate


@dataclass
class PowerMetricsResult:
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

# Incremental regression class for timestamp conversion
# source: https://github.com/wolfpld/tracy, tracy/public/tracy/TracyCUDA.hpp

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
    # pmd2-cli can terminate with a partially written final line.
    # Skip malformed rows and keep valid samples.
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = _normalize_columns(df)
    required = ["timestamp_ns", "total_power_w", "sensor4_power_mw", "sensor7_power_mw"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"PMD2 CSV missing required columns: {', '.join(missing)}")
    df["timestamp_ns"] = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    df = df.dropna(subset=["timestamp_ns"]).copy()
    if df.empty:
        raise ValueError("PMD2 CSV has no valid timestamp rows")
    t0 = df["timestamp_ns"].iloc[0]
    df["t_s"] = (df["timestamp_ns"] - t0) / 1_000_000_000.0
    df["total_power_w"] = pd.to_numeric(df["total_power_w"], errors="coerce")
    df["sensor4_power_w"] = pd.to_numeric(df["sensor4_power_mw"], errors="coerce") / 1000.0
    df["sensor7_power_w"] = pd.to_numeric(df["sensor7_power_mw"], errors="coerce") / 1000.0
    df = df.dropna(subset=["t_s", "total_power_w", "sensor4_power_w", "sensor7_power_w"]).copy()
    if df.empty:
        raise ValueError("PMD2 CSV has no valid power rows")
    return df

def gpu_to_cpu_time(gpu_timestamp: float, slope: float, intercept: float) -> float:
    """Convert GPU timestamp to CPU timestamp using linear regression."""
    return slope * gpu_timestamp + intercept


def _parse_output_exports(exports: Any) -> dict[str, Any]:
    """Normalize metrics fields from output.json exports."""
    raw_offsets = exports.get("timestamp_offsets", [])
    offsets: list[tuple[float, float]] = []
    for sample in raw_offsets:
        try:
            gpu_ts = float(sample["cupti_timestamp"])
            cpu_initial = float(sample["cpu_initial_timestamp"])
            cpu_final = float(sample["cpu_final_timestamp"])
            offsets.append((gpu_ts, (cpu_initial + cpu_final) / 2.0))
        except Exception:
            continue

    kernel_start = float(
        exports.get("start_timestamp", exports.get("kernel_start", exports.get("kernel_start_timestamp", 0.0)))
    )
    kernel_end = float(
        exports.get("end_timestamp", exports.get("kernel_end", exports.get("kernel_end_timestamp", 0.0)))
    )
    kernel_duration = float(
        exports.get("duration", exports.get("kernel_duration", max(kernel_end - kernel_start, 0.0)))
    )

    if kernel_end <= kernel_start and kernel_duration > 0.0:
        kernel_end = kernel_start + kernel_duration

    return {
        "offsets": offsets,
        "kernel_start": kernel_start,
        "kernel_end": kernel_end,
        "kernel_duration": kernel_duration,
    }


def extract_power_metrics(path: Path, exports: Any) -> PowerMetricsResult:
    """Extract power metrics from GPU data.

    Args:
        path: Path to the directory containing CSV files

    Returns:
        PowerMetrics object containing calculated metrics
    """
    # Parse timing information from output.json exports
    output_info = _parse_output_exports(exports)
    if output_info["kernel_duration"] <= 0.0:
        raise ValueError(f"missing/invalid kernel timing fields in exports: {sorted(exports.keys())}")
    if len(output_info["offsets"]) < 2:
        raise ValueError("insufficient timestamp_offsets samples for regression")

    pmd2_path = path / "pmd2.csv"
    if not pmd2_path.exists() or pmd2_path.stat().st_size == 0:
        raise FileNotFoundError(f"pmd2.csv missing or empty at {pmd2_path}")
    try:
        pmd2 = load_pmd2(pmd2_path)
    except Exception as exc:
        raise RuntimeError(f"unable to parse PMD2 CSV: {exc}") from exc

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

    # Convert to seconds for processing
    kernel_start_cpu_s = (kernel_start_cpu_ns - pmd2_t0_ns) / 1e9
    kernel_end_cpu_s = (kernel_end_cpu_ns - pmd2_t0_ns) / 1e9
    kernel_duration_cpu_s = kernel_end_cpu_s - kernel_start_cpu_s

    # Calculate energy consumed during kernel execution using trapezoidal integration
    kernel_mask = (pmd2["t_s"] >= kernel_start_cpu_s) & (pmd2["t_s"] <= kernel_end_cpu_s)
    pmd2_kernel = pmd2[kernel_mask].copy()
    print(f"  [power] kernel window: {kernel_start_cpu_s:.3f}s – {kernel_end_cpu_s:.3f}s  ({kernel_duration_cpu_s*1000:.1f} ms)", flush=True)
    print(f"  [power] pmd2 total={len(pmd2)} samples, in-window={len(pmd2_kernel)}", flush=True)
    
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

    return PowerMetricsResult(
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
