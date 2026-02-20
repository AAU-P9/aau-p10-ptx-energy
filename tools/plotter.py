#!/usr/bin/env python3
"""Plot combined GPU power data from nvidia-smi and pmd2 CSVs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


NVIDIA_DEFAULT = Path("build/example/nvidia-smi.csv")
PMD2_DEFAULT = Path("build/example/pmd2.csv")


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
    df["timestamp_us"] = pd.to_numeric(df["timestamp_us"], errors="coerce")
    df = df.dropna(subset=["timestamp_us"]).copy()
    t0 = df["timestamp_us"].iloc[0]
    df["t_s"] = (df["timestamp_us"] - t0) / 1_000_000.0
    df["total_power_w"] = pd.to_numeric(df["total_power_w"], errors="coerce")
    df["sensor4_power_w"] = pd.to_numeric(df["sensor4_power_mw"], errors="coerce") / 1000.0
    df["sensor7_power_w"] = pd.to_numeric(df["sensor7_power_mw"], errors="coerce") / 1000.0
    return df


def build_figure(nv: pd.DataFrame, pmd2: pd.DataFrame) -> go.Figure:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot combined CSV data with Plotly")
    parser.add_argument("path", type=Path, help="Path to the directory containing CSV files")
    args = parser.parse_args()

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"plots/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    nv_path = args.path / "nvidia-smi.csv"
    pmd2_path = args.path / "pmd2.csv"
    nv = load_nvidia_smi(nv_path)
    pmd2 = load_pmd2(pmd2_path)
    
    # Create separate figures for power and temperature
    power_fig = build_figure(nv, pmd2)
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
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
