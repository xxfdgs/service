#!/usr/bin/env python3
"""Plot the strongest existing evidence of input-to-feedback domain shift."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "results/train_feedback_domain_audit"
OUT_DIR = AUDIT_DIR / "figures"
INPUT_PATH = ROOT / "results/deduplicated_rebaseline/data_audit/dataset_with_sample_id.csv"
FEEDBACK_PATH = ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv"

TARGETS = [
    "EE_before",
    "EE_after",
    "Aerosolization_Efficiency",
    "mRNA_Recovery_Efficiency",
    "Norm_before",
    "Norm_after",
]
TARGET_LABELS = {
    "EE_before": "EE before",
    "EE_after": "EE after",
    "Aerosolization_Efficiency": "Aerosolization",
    "mRNA_Recovery_Efficiency": "mRNA recovery",
    "Norm_before": "Norm before",
    "Norm_after": "Norm after",
}
CORE_TARGETS = TARGETS[:4]
INPUT_COLOR = "#2878B5"
FEEDBACK_COLOR = "#D94E4E"
ACCENT_COLOR = "#F2A104"
GRID_COLOR = "#D9DEE7"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pretty_feature(space: str, feature: str) -> str:
    descriptor_names = {
        0: "SsNH3",
        1: "SMR VSA9",
        2: "SlogP VSA11",
        3: "SlogP VSA10",
        4: "TopoPSA",
        5: "MW",
        6: "nRot",
        7: "nRing",
        8: "nAromAtom",
        9: "nHBDon",
        10: "nHBAcc",
    }
    if space == "raw_11d_descriptor" and feature.startswith("component_"):
        parts = feature.split("_")
        component = int(parts[1])
        index = int(parts[-1])
        return f"C{component} {descriptor_names.get(index, f'desc {index}')}"
    replacements = {
        "ratio_component_": "Ratio C",
        "ratio_min_nonzero": "Minimum nonzero ratio",
        "ratio_max": "Maximum ratio",
        "ratio_sum": "Ratio sum",
        "formulation_entropy": "Formulation entropy",
        "effective_component_count": "Effective component count",
    }
    for old, new in replacements.items():
        if feature == old:
            return new
        if feature.startswith(old):
            return feature.replace(old, new)
    return feature.replace("_", " ")


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    return x, np.arange(1, len(x) + 1) / len(x)


def plot_overview(input_frame: pd.DataFrame, feedback_frame: pd.DataFrame) -> None:
    domain = pd.read_csv(AUDIT_DIR / "domain_classifier_metrics.csv")
    summary = pd.read_csv(AUDIT_DIR / "dataset_summary.csv")
    features = pd.read_csv(AUDIT_DIR / "feature_shift_metrics.csv")
    errors = pd.read_csv(AUDIT_DIR / "subgroup_error_metrics.csv")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.suptitle(
        "Input vs feedback: evidence of domain shift and degraded generalization",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    # A: out-of-fold domain classification.
    ax = axes[0, 0]
    names = domain["model"].replace({"LogisticRegression": "Logistic regression"})
    bars = ax.barh(names, domain["roc_auc"], color=[INPUT_COLOR, ACCENT_COLOR], height=0.55)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1.4, label="No separation (AUC=0.5)")
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("5-fold CV ROC AUC")
    ax.set_title("A  Dataset origin is highly predictable")
    for bar, value in zip(bars, domain["roc_auc"]):
        ax.text(value - 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="right", va="center", color="white", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    # B: component/formulation novelty.
    ax = axes[0, 1]
    coverage = summary.loc[summary.dataset.eq("feedback_coverage")].iloc[0]
    novelty = np.array([coverage["new_component_fraction"], coverage["new_formula_fraction"]], float)
    labels = ["Contains unseen\ncomponent", "Unseen formulation"]
    ax.bar(labels, 1 - novelty, color="#B8C6D9", label="Seen in input")
    ax.bar(labels, novelty, bottom=1 - novelty, color=FEEDBACK_COLOR, label="Unseen in input")
    for index, value in enumerate(novelty):
        ax.text(index, 1 - value / 2, f"{value:.1%}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of feedback samples")
    ax.set_title("B  Most feedback chemistry is novel")
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    # C: strongest feature shifts.
    ax = axes[0, 2]
    top = (
        features.loc[features.feature_type.eq("numeric")]
        .sort_values("psi", ascending=False)
        .drop_duplicates(["feature_space", "feature"])
        .head(8)
        .sort_values("psi")
        .copy()
    )
    labels = [pretty_feature(space, feature) for space, feature in zip(top.feature_space, top.feature)]
    colors = [INPUT_COLOR if space == "F2" else ACCENT_COLOR for space in top.feature_space]
    bars = ax.barh(labels, top.psi, color=colors)
    ax.axvline(0.25, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("C  Largest input-feature shifts")
    for bar, psi_value, ks_value in zip(bars, top.psi, top.ks_statistic):
        ax.text(psi_value + 0.12, bar.get_y() + bar.get_height() / 2, f"KS={ks_value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(top.psi) * 1.24)

    # D: label mean shift in input standard deviations, all six targets.
    ax = axes[1, 0]
    shifts = []
    for target in TARGETS:
        train_values = pd.to_numeric(input_frame[target], errors="coerce")
        feedback_values = pd.to_numeric(feedback_frame[target], errors="coerce")
        shifts.append((feedback_values.mean() - train_values.mean()) / train_values.std(ddof=1))
    y = np.arange(len(TARGETS))
    colors = [FEEDBACK_COLOR if abs(value) >= 0.5 else INPUT_COLOR for value in shifts]
    ax.barh(y, shifts, color=colors)
    ax.set_yticks(y, [TARGET_LABELS[target] for target in TARGETS])
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axvline(-0.5, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("(feedback mean - input mean) / input SD")
    ax.set_title("D  Target distributions also shift")
    for index, value in enumerate(shifts):
        if abs(value) >= 0.35:
            ax.text(value / 2, index, f"{value:+.2f}", va="center", ha="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(value + (0.03 if value >= 0 else -0.03), index, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)

    # E: feedback error relative to strictly out-of-fold internal error.
    ax = axes[1, 1]
    id_rows = errors.loc[
        errors.subgroup_type.eq("ood_class") & errors.subgroup.eq("ID") & errors.target.isin(CORE_TARGETS)
    ]
    error_pivot = id_rows.pivot(index="target", columns="dataset", values="mae").loc[CORE_TARGETS]
    ratios = error_pivot["feedback"] / error_pivot["internal_oof_test"]
    bars = ax.barh(
        np.arange(len(CORE_TARGETS)),
        ratios,
        color=[FEEDBACK_COLOR if value > 1.2 else INPUT_COLOR for value in ratios],
    )
    ax.set_yticks(np.arange(len(CORE_TARGETS)), [TARGET_LABELS[target] for target in CORE_TARGETS])
    ax.invert_yaxis()
    ax.axvline(1, color="#333333", linewidth=1.2, label="Same MAE")
    ax.axvline(1.2, color="#777777", linestyle="--", linewidth=1.2, label="20% higher")
    ax.set_xlabel("Feedback MAE / internal OOF MAE")
    ax.set_title("E  Feedback error is substantially higher")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for bar, value in zip(bars, ratios):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}×", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(ratios) * 1.18)

    # F: prediction variance collapse with increasing feedback distance.
    ax = axes[1, 2]
    compression = errors.loc[
        errors.dataset.eq("feedback")
        & errors.subgroup_type.eq("ood_distance_tertile")
        & errors.target.isin(CORE_TARGETS)
    ].copy()
    order = ["low", "middle", "high"]
    for target, color, marker in zip(CORE_TARGETS, [INPUT_COLOR, ACCENT_COLOR, "#6B5CA5", FEEDBACK_COLOR], ["o", "s", "^", "D"]):
        part = compression.loc[compression.target.eq(target)].set_index("subgroup").loc[order]
        ax.plot(order, part.prediction_to_label_std_ratio, marker=marker, linewidth=2, markersize=6, color=color, label=TARGET_LABELS[target])
    ax.axhline(1, color="#333333", linestyle="--", linewidth=1.2, label="No variance compression")
    ax.set_xlabel("Feedback distance tertile")
    ax.set_ylabel("Prediction SD / label SD")
    ax.set_title("F  Predictions collapse toward the mean")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    fig.text(
        0.5,
        0.012,
        "Domain classifier uses F2 features with out-of-fold predictions. Error and compression panels use the locked tree baseline; no feedback sample was used for model selection.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=2.2, w_pad=1.8)
    save_figure(fig, "input_feedback_domain_shift_overview")


def plot_ecdf_grid(
    input_frame: pd.DataFrame,
    feedback_frame: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    stem: str,
    title: str,
    shape: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(*shape, figsize=(14, 10 if shape[0] == 3 else 8))
    axes = np.asarray(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        train_x, train_y = ecdf(input_frame[column])
        feedback_x, feedback_y = ecdf(feedback_frame[column])
        ax.step(train_x, train_y, where="post", color=INPUT_COLOR, linewidth=2.2, label="Input")
        ax.step(feedback_x, feedback_y, where="post", color=FEEDBACK_COLOR, linewidth=2.2, label="Feedback")
        train_mean = np.mean(train_x)
        feedback_mean = np.mean(feedback_x)
        ax.axvline(train_mean, color=INPUT_COLOR, linestyle=":", linewidth=1.4)
        ax.axvline(feedback_mean, color=FEEDBACK_COLOR, linestyle=":", linewidth=1.4)
        ax.set_title(labels[column])
        ax.set_xlabel("Value")
        ax.set_ylabel("Cumulative fraction")
        ax.grid(axis="both")
        ax.text(
            0.98,
            0.04,
            f"mean: {train_mean:.2f} → {feedback_mean:.2f}\n"
            f"valid n: {len(train_x)} → {len(feedback_x)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    for ax in axes[len(columns) :]:
        ax.axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.0, w_pad=1.5)
    save_figure(fig, stem)


def main() -> None:
    setup_style()
    input_frame = pd.read_csv(INPUT_PATH)
    feedback_frame = pd.read_csv(FEEDBACK_PATH)
    plot_overview(input_frame, feedback_frame)
    plot_ecdf_grid(
        input_frame,
        feedback_frame,
        TARGETS,
        TARGET_LABELS,
        "input_feedback_target_ecdf",
        "Target-distribution shift between input and feedback",
        (3, 2),
    )
    ratio_columns = ["mol%_IL", "mol%_HL", "mol%_Chol", "mol%_PEG", "mol%_Fifth"]
    ratio_labels = {
        "mol%_IL": "Ionizable lipid ratio",
        "mol%_HL": "Helper lipid ratio",
        "mol%_Chol": "Cholesterol ratio",
        "mol%_PEG": "PEG-lipid ratio",
        "mol%_Fifth": "Fifth-component ratio",
    }
    plot_ecdf_grid(
        input_frame,
        feedback_frame,
        ratio_columns,
        ratio_labels,
        "input_feedback_ratio_ecdf",
        "Formulation-ratio shift between input and feedback",
        (2, 3),
    )
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
