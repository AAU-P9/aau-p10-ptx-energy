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
    df["total_power_w"] = pd.to_numeric(df["total_power_mw"], errors="coerce") / 1000.0
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
        fig.add_trace(
            go.Scatter(
                x=nv["t_s"],
                y=nv["temperature_c"],
                mode="lines",
                name="nvidia-smi temperature [C]",
                yaxis="y2",
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
    parser.add_argument("--nvidia", type=Path, default=NVIDIA_DEFAULT)
    parser.add_argument("--pmd2", type=Path, default=PMD2_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path("build/example/combined_plot.html"))
    args = parser.parse_args()

    nv = load_nvidia_smi(args.nvidia)
    pmd2 = load_pmd2(args.pmd2)
    fig = build_figure(nv, pmd2)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
