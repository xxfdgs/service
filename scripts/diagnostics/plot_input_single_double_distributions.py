#!/usr/bin/env python3
"""Plot each 20260812 target distribution for single versus double Fifths.

Only rows explicitly labelled ``single`` or ``double`` are included.  Each
target receives its own density-normalised histogram/KDE figure with identical
bin edges for the two classes, plus a CSV of the underlying summary statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after",
]
CLASS_STYLE = {
    "single": {"color": "#4c78a8", "label": "single"},
    "double": {"color": "#f58518", "label": "double"},
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read the historical input CSV without changing the source encoding."""
    failures: list[str] = []
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            failures.append(f"{encoding}: {error}")
    raise UnicodeDecodeError("input", b"", 0, 1, "; ".join(failures))


def bin_edges(values: np.ndarray, bin_width: float = 0.1) -> np.ndarray:
    """Return shared, narrow histogram bins aligned to integer endpoints."""
    if bin_width <= 0 or not np.isclose(1.0 / bin_width, round(1.0 / bin_width)):
        raise ValueError(
            "bin_width must be a positive divisor of 1, e.g. 1.0, 0.5 or 0.25"
        )

    lower = np.floor(float(values.min()) / bin_width) * bin_width
    upper = np.ceil(float(values.max()) / bin_width) * bin_width

    if np.isclose(lower, upper):
        lower -= bin_width
        upper += bin_width

    return np.arange(lower, upper + bin_width * 0.5, bin_width)


def plot_target(frame: pd.DataFrame, target: str, output_dir: Path) -> list[dict[str, object]]:
    values_by_class = {
        name: frame.loc[frame["fifth_class"].eq(name), target].dropna().to_numpy(dtype=float)
        for name in CLASS_STYLE
    }
    combined = np.concatenate([values for values in values_by_class.values() if len(values)])
    if len(combined) < 2:
        raise ValueError(f"Not enough labelled {target} values to plot.")
    edges = bin_edges(combined)
    lower, upper = float(edges[0]), float(edges[-1])
    padding = max((upper - lower) * 0.04, 0.05)
    x_grid = np.linspace(lower - padding, upper + padding, 512)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    summary: list[dict[str, object]] = []
    for name, style in CLASS_STYLE.items():
        values = values_by_class[name]
        if not len(values):
            continue
        axis.hist(values, bins=edges, density=True, alpha=0.30, color=style["color"],
                  edgecolor=style["color"], linewidth=0.8)
        if len(values) > 1 and np.std(values) > 0:
            axis.plot(x_grid, gaussian_kde(values)(x_grid), color=style["color"], linewidth=2.2,
                      label=f"{style['label']} (n={len(values)}, mean={values.mean():.2f})")
        else:
            axis.axvline(values[0], color=style["color"], linewidth=2.2,
                        label=f"{style['label']} (n={len(values)})")
        summary.append({
            "target": target,
            "Fifth_class": name,
            "n": len(values),
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values, ddof=0)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    axis.set(title=f"20260812-sum-700: {target}", xlabel=target, ylabel="Density")
    axis.set_xlim(lower - padding, upper + padding)
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, fontsize=9)
    figure.savefig(output_dir / f"{target}_single_vs_double_distribution.png", dpi=220)
    figure.savefig(output_dir / f"{target}_single_vs_double_distribution.pdf")
    plt.close(figure)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path,
                        default=root / "datasets_lrx/raw/input/20260812-sum-700.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/input_graphgps_optimization/data_distribution_20260812_single_double")
    args = parser.parse_args()
    source = args.input_csv.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = read_csv(source)
    required = {"Fifth_class", *TARGETS}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Input CSV misses required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["fifth_class"] = frame["Fifth_class"].fillna("").astype(str).str.strip().str.lower()
    frame = frame.loc[frame["fifth_class"].isin(CLASS_STYLE)].copy()
    if frame.empty:
        raise ValueError("No rows labelled single or double.")
    statistics = []
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        statistics.extend(plot_target(frame, target, output))
    summary = pd.DataFrame(statistics)
    summary.to_csv(output / "single_double_target_distribution_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote six distribution figures to: {output}")


if __name__ == "__main__":
    main()
