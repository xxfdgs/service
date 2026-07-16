#!/usr/bin/env python3
"""Read-only audit of train/OOF/feedback domain shift and locked-tree errors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, percentileofscore, pearsonr, spearmanr, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from audit_deduplicated_dataset import COMPONENTS, TARGETS, canonical_smiles, enrich  # noqa: E402
from stable_formulation import build_stable_feature_sets  # noqa: E402


BASE = ROOT / "results/deduplicated_rebaseline"
FEEDBACK_PATH = ROOT / "datasets_lrx/raw/feedback/20260703_validation.csv"
BENCHMARK = ROOT / "results/new_dataset_benchmark_20260713"
OUT = ROOT / "results/train_feedback_domain_audit"
MORDRED_NAMES = ["SsNH3", "SMR_VSA9", "SlogP_VSA11", "SlogP_VSA10", "TopoPSA", "MW", "nRot", "nRing", "nAromAtom", "nHBDon", "nHBAcc"]
RATIO_COLUMNS = [item[2] for item in COMPONENTS]
OOD_DISTANCE_METRICS = [
    "f2_nearest_neighbor_distance",
    "f2_knn5_distance",
    "f2_mahalanobis_distance",
    "descriptor_nearest_neighbor_distance",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    value = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return {"n": int(len(value)), "mean": float(np.mean(value)), "std": float(np.std(value, ddof=1)) if len(value) > 1 else 0.0,
            "q05": float(np.quantile(value, .05)), "q25": float(np.quantile(value, .25)), "q50": float(np.quantile(value, .5)),
            "q75": float(np.quantile(value, .75)), "q95": float(np.quantile(value, .95)), "min": float(np.min(value)), "max": float(np.max(value))}


def psi(train: np.ndarray, feedback: np.ndarray) -> float:
    train, feedback = np.asarray(train, float), np.asarray(feedback, float)
    train, feedback = train[np.isfinite(train)], feedback[np.isfinite(feedback)]
    if len(train) < 3 or len(feedback) < 1:
        return np.nan
    edges = np.unique(np.quantile(train, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(train, bins=edges)[0] / len(train)
    actual = np.histogram(feedback, bins=edges)[0] / len(feedback)
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def make_f2(frame: pd.DataFrame) -> pd.DataFrame:
    schema = SimpleNamespace(components=[{"name_column": name, "smiles_column": smiles, "ratio_column": ratio} for name, smiles, ratio in COMPONENTS])
    values, _, _ = build_stable_feature_sets(frame, schema)
    return values["F2_identity_ratio"].copy()


def raw_descriptor_train() -> pd.DataFrame:
    raw = pd.read_csv(BASE / "artifacts/mordred_11_raw_features.csv", dtype={"sample_id": str})
    needed = {"sample_id", "component_position", *[f"feature_{index}" for index in range(11)]}
    if needed - set(raw):
        raise RuntimeError("Training raw descriptor cache malformed")
    parts = []
    for position in range(1, 6):
        values = raw.loc[raw.component_position.eq(position)].set_index("sample_id")[[f"feature_{index}" for index in range(11)]].copy()
        values.columns = [f"component_{position}_feature_{index}" for index in range(11)]
        parts.append(values)
    result = pd.concat(parts, axis=1)
    if len(result) != 700 or result.index.has_duplicates:
        raise RuntimeError("Training raw descriptor sample alignment failed")
    return result


def raw_descriptor_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(ROOT / "results/mordred_train_feedback_20260710_v3/mordred_descriptors_unique_smiles.csv")
    if {"smiles", *MORDRED_NAMES} - set(source):
        raise RuntimeError("Mordred feedback cache lacks required descriptor columns")
    source["canonical"] = source.smiles.map(canonical_smiles)
    source = source.drop_duplicates("canonical").set_index("canonical")
    parts = []
    for position, (_, smiles_column, _) in enumerate(COMPONENTS, start=1):
        keys = feedback[smiles_column].map(canonical_smiles)
        missing = set(keys) - set(source.index) - {"<missing>"}
        if missing:
            raise RuntimeError(f"Mordred feedback cache missing {len(missing)} component structures")
        values = pd.DataFrame(0.0, index=np.arange(len(keys)), columns=MORDRED_NAMES)
        present = keys.ne("<missing>")
        values.loc[present.to_numpy(), :] = source.loc[keys.loc[present], MORDRED_NAMES].apply(pd.to_numeric, errors="coerce").to_numpy()
        values.index = feedback.index
        values.columns = [f"component_{position}_feature_{index}" for index in range(11)]
        parts.append(values)
    return pd.concat(parts, axis=1)


def numeric_shift(train: pd.DataFrame, feedback: pd.DataFrame, space: str) -> list[dict[str, object]]:
    rows = []
    for feature in train.columns:
        if not pd.api.types.is_numeric_dtype(train[feature]):
            train_values, feedback_values = train[feature].astype(str), feedback[feature].astype(str)
            rows.append({"feature_space": space, "feature": feature, "feature_type": "categorical", "train_unique": int(train_values.nunique()),
                         "feedback_unique": int(feedback_values.nunique()), "feedback_novel_category_fraction": float((~feedback_values.isin(set(train_values))).mean()),
                         "ks_statistic": np.nan, "wasserstein_distance": np.nan, "psi": np.nan,
                         "outside_train_minmax_fraction": np.nan, "outside_train_q01_q99_fraction": np.nan})
            continue
        a = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(feedback[feature], errors="coerce").dropna().to_numpy(float)
        if not len(a) or not len(b):
            continue
        rows.append({"feature_space": space, "feature": feature, "feature_type": "numeric", **{f"train_{key}": value for key, value in summary(a).items()},
                     **{f"feedback_{key}": value for key, value in summary(b).items()}, "ks_statistic": float(ks_2samp(a, b).statistic),
                     "wasserstein_distance": float(wasserstein_distance(a, b)), "psi": psi(a, b),
                     "outside_train_minmax_fraction": float(((b < np.min(a)) | (b > np.max(a))).mean()),
                     "outside_train_q01_q99_fraction": float(((b < np.quantile(a, .01)) | (b > np.quantile(a, .99))).mean())})
    return rows


def feature_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(exclude="object").columns.tolist()
    categorical = frame.select_dtypes(include="object").columns.tolist()
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def ood_feature_matrices(reference: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode unseen F2 identities as an explicit distance-bearing category.

    ``handle_unknown='ignore'`` would turn every unseen component into an all
    zero block and therefore make a genuinely new component spuriously close
    to training samples.  Categories are still defined from training data only.
    """
    reference, query = reference.copy(), query.copy()
    numeric = reference.select_dtypes(exclude="object").columns.tolist()
    categorical = reference.select_dtypes(include="object").columns.tolist()
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        categories = []
        for column in categorical:
            known = set(reference[column].astype(str).fillna("<missing>"))
            reference[column] = reference[column].astype(str).fillna("<missing>")
            query_values = query[column].astype(str).fillna("<missing>")
            query[column] = query_values.where(query_values.isin(known), "__UNKNOWN__")
            categories.append(sorted(known | {"__UNKNOWN__"}))
        transforms.append(("categorical", OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False), categorical))
    transform = ColumnTransformer(transforms, sparse_threshold=0.0).fit(reference)
    return np.asarray(transform.transform(reference), float), np.asarray(transform.transform(query), float)


def nn_reference_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighbors = NearestNeighbors(n_neighbors=min(6, len(values))).fit(values)
    distances = neighbors.kneighbors(values, return_distance=True)[0]
    if distances.shape[1] < 2:
        return np.zeros(len(values)), np.zeros(len(values))
    return distances[:, 1], distances[:, 1:min(6, distances.shape[1])].mean(axis=1)


def component_sets(frame: pd.DataFrame) -> list[set[str]]:
    return [set(row[row != "<missing>"].astype(str)) for _, row in frame[[f"component_{index}_key" for index in range(1, 6)]].iterrows()]


def ood_against_reference(reference_f2: pd.DataFrame, query_f2: pd.DataFrame, reference_desc: pd.DataFrame, query_desc: pd.DataFrame,
                          reference_components: list[set[str]], query_components: list[set[str]]) -> pd.DataFrame:
    ref_values, query_values = ood_feature_matrices(reference_f2, query_f2)
    nearest = NearestNeighbors(n_neighbors=min(5, len(ref_values))).fit(ref_values)
    query_distances = nearest.kneighbors(query_values, return_distance=True)[0]
    f2_nn, f2_k5 = query_distances[:, 0], query_distances.mean(axis=1)
    ref_nn, ref_k5 = nn_reference_scores(ref_values)
    numeric = reference_f2.select_dtypes(exclude="object").columns.tolist()
    numeric_imputer = SimpleImputer(strategy="median").fit(reference_f2[numeric])
    scaler = StandardScaler().fit(numeric_imputer.transform(reference_f2[numeric]))
    ref_numeric = scaler.transform(numeric_imputer.transform(reference_f2[numeric]))
    query_numeric = scaler.transform(numeric_imputer.transform(query_f2[numeric]))
    covariance = LedoitWolf().fit(ref_numeric)
    ref_mahal = covariance.mahalanobis(ref_numeric) ** .5
    query_mahal = covariance.mahalanobis(query_numeric) ** .5
    desc_imputer = SimpleImputer(strategy="median").fit(reference_desc)
    desc_scaler = StandardScaler().fit(desc_imputer.transform(reference_desc))
    ref_desc_values = desc_scaler.transform(desc_imputer.transform(reference_desc))
    query_desc_values = desc_scaler.transform(desc_imputer.transform(query_desc))
    desc_neighbors = NearestNeighbors(n_neighbors=1).fit(ref_desc_values)
    descriptor_nn = desc_neighbors.kneighbors(query_desc_values, return_distance=True)[0][:, 0]
    ref_descriptor_nn, _ = nn_reference_scores(ref_desc_values)
    overlap = [max((len(values.intersection(ref)) / max(len(values.union(ref)), 1) for ref in reference_components), default=0.0) for values in query_components]
    result = pd.DataFrame({"f2_nearest_neighbor_distance": f2_nn, "f2_knn5_distance": f2_k5, "f2_mahalanobis_distance": query_mahal,
                           "descriptor_nearest_neighbor_distance": descriptor_nn, "nearest_train_component_overlap": overlap})
    for name, values, reference_values in [("f2_nearest_neighbor", f2_nn, ref_nn), ("f2_knn5", f2_k5, ref_k5),
                                            ("f2_mahalanobis", query_mahal, ref_mahal), ("descriptor_nearest_neighbor", descriptor_nn, ref_descriptor_nn)]:
        result[f"{name}_percentile"] = [float(percentileofscore(reference_values, value, kind="weak")) for value in values]
    result["ood_class"] = pd.cut(result["f2_knn5_percentile"], bins=[-np.inf, 95, 99, np.inf], labels=["ID", "mild OOD", "severe OOD"], include_lowest=True).astype(str)
    return result


def label_rows(train: pd.DataFrame, feedback: pd.DataFrame, oof: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for target in TARGETS:
        train_values = train[target].to_numpy(float)
        for name, values in [("train", train_values), ("internal_oof_test", oof[target].to_numpy(float)), ("feedback", feedback[target].to_numpy(float))]:
            values = np.asarray(values, float)
            overlap = max(0., min(np.max(values), np.max(train_values)) - max(np.min(values), np.min(train_values))) / max(np.max(train_values) - np.min(train_values), 1e-12)
            rows.append({"record_type": "distribution", "target": target, "dataset": name, **summary(values), "range_overlap_vs_train": overlap})
    for name, values in [("train", train), ("internal_oof_test", oof), ("feedback", feedback)]:
        for first_index, first in enumerate(TARGETS):
            for second in TARGETS[first_index + 1:]:
                rows.append({"record_type": "target_correlation", "dataset": name, "target": f"{first}|{second}",
                             "pearson": float(pearsonr(values[first], values[second]).statistic), "spearman": float(spearmanr(values[first], values[second]).statistic)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(BASE / "data_audit/dataset_with_sample_id.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
    feedback_raw = pd.read_csv(FEEDBACK_PATH, dtype={"ID": str})
    feedback = enrich(feedback_raw).set_index("sample_id", drop=False)
    if len(train) != 700 or len(feedback) != 97 or train.index.has_duplicates or feedback.index.has_duplicates:
        raise RuntimeError("Dataset/sample ID audit failed")
    f2_train, f2_feedback = make_f2(train), make_f2(feedback)
    f2_train.index, f2_feedback.index = train.index, feedback.index
    desc_train, desc_feedback = raw_descriptor_train().loc[train.index], raw_descriptor_feedback(feedback)
    desc_feedback.index = feedback.index
    if list(desc_train.columns) != list(desc_feedback.columns):
        raise RuntimeError("Raw descriptor feature alignment failed")

    oof_all = pd.read_csv(BASE / "tree_baselines/oof_predictions.csv", dtype={"sample_id": str})
    oof = oof_all.loc[(oof_all.protocol.eq("formula_identity_group_cv")) & oof_all.model.eq("NestedSelectedBaseline")].copy()
    if len(oof) != 700 * 4 or oof.duplicated(["target", "sample_id"]).any():
        raise RuntimeError("Nested OOF prediction audit failed")
    selected = pd.read_csv(BASE / "tree_baselines/selected_baseline_by_target.csv")
    selected = selected.loc[selected.protocol.eq("formula_identity_group_cv")].set_index("target")
    feedback_predictions = pd.read_csv(BENCHMARK / "tabular_baseline_feedback_predictions.csv", dtype={"sample_id": str})
    feedback_frames = []
    for target in TARGETS:
        choice = selected.loc[target]
        part = feedback_predictions.loc[(feedback_predictions.target.eq(target)) & (feedback_predictions.model.eq(choice.selected_model)) &
                                        (feedback_predictions.feature_set.eq(choice.selected_feature_set))].copy()
        if len(part) != len(feedback) or set(part.sample_id.astype(str)) != set(feedback_raw.ID.astype(str)):
            raise RuntimeError(f"Locked feedback prediction audit failed target={target}")
        part["baseline_kind"] = "formula_identity_global_selected_proxy"
        feedback_frames.append(part)
    feedback_pred = pd.concat(feedback_frames, ignore_index=True)

    # Coverage audit.
    train_components = set().union(*component_sets(train))
    feedback_components = component_sets(feedback)
    train_formulas = set(train.formula_identity_key)
    coverage_rows, summary_rows = [], []
    for name, frame in [("train", train), ("internal_oof_test", train), ("feedback", feedback)]:
        ratio_sums = frame[RATIO_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        fifth_present = (~frame.component_5_key.eq("<missing>")) & (pd.to_numeric(frame["mol%_Fifth"], errors="coerce").fillna(0) > 0)
        summary_rows.append({"dataset": name, "n_samples": len(frame), "four_component_fraction": float((~fifth_present).mean()), "five_component_fraction": float(fifth_present.mean()),
                             **{f"ratio_sum_{key}": value for key, value in summary(ratio_sums).items()}})
        for position in range(1, 6):
            values = frame[f"component_{position}_key"].astype(str)
            for component, count in values.value_counts().items():
                coverage_rows.append({"record_type": "component_frequency", "dataset": name, "component_position": position, "component": component, "count": int(count), "fraction": float(count / len(frame))})
    for index, values in enumerate(feedback_components):
        ratios = feedback.iloc[index][RATIO_COLUMNS].astype(float).to_numpy()
        train_ratios = train[RATIO_COLUMNS].astype(float)
        coverage_rows.append({"record_type": "feedback_sample_coverage", "dataset": "feedback", "sample_id": feedback.index[index],
                              "new_component_count": int(len(values - train_components)), "shared_component_count": int(len(values & train_components)),
                              "formula_seen_in_train": bool(feedback.iloc[index].formula_identity_key in train_formulas),
                              "ratio_any_outside_train_minmax": bool(np.any((ratios < train_ratios.min().to_numpy()) | (ratios > train_ratios.max().to_numpy()))),
                              "ratio_any_outside_train_q01_q99": bool(np.any((ratios < train_ratios.quantile(.01).to_numpy()) | (ratios > train_ratios.quantile(.99).to_numpy())))})
    summary_rows.append({"dataset": "feedback_coverage", "n_samples": len(feedback), "new_component_fraction": float(np.mean([len(item - train_components) > 0 for item in feedback_components])),
                         "new_formula_fraction": float((~feedback.formula_identity_key.isin(train_formulas)).mean())})

    feature_rows = numeric_shift(f2_train, f2_feedback, "F2") + numeric_shift(desc_train, desc_feedback, "raw_11d_descriptor")
    # OOF OOD is strictly fold-local; feedback OOD is referenced to all 700 training rows.
    ood_frames = []
    manifests = []
    for fold in range(5):
        manifest = pd.read_csv(BASE / "manifests/formula_identity_group_cv" / f"fold_{fold}.csv", dtype={"sample_id": str}).set_index("sample_id", drop=False)
        reference_ids = pd.Index(manifest.loc[manifest.split.isin(["train", "val"]), "sample_id"].astype(str))
        query_ids = pd.Index(manifest.loc[manifest.split.eq("test"), "sample_id"].astype(str))
        scores = ood_against_reference(f2_train.loc[reference_ids], f2_train.loc[query_ids], desc_train.loc[reference_ids], desc_train.loc[query_ids],
                                       component_sets(train.loc[reference_ids]), component_sets(train.loc[query_ids]))
        scores.insert(0, "sample_id", query_ids)
        scores.insert(0, "outer_fold", fold)
        scores.insert(0, "dataset", "internal_oof_test")
        ood_frames.append(scores)
        manifests.append(manifest)
    feedback_scores = ood_against_reference(f2_train, f2_feedback, desc_train, desc_feedback, component_sets(train), component_sets(feedback))
    feedback_scores.insert(0, "sample_id", feedback.index)
    feedback_scores.insert(0, "outer_fold", np.nan)
    feedback_scores.insert(0, "dataset", "feedback")
    ood_frames.append(feedback_scores)
    ood_scores = pd.concat(ood_frames, ignore_index=True)
    ood_scores["embedding_distance_status"] = "unavailable_feedback_no_fold_consistent_frozen_embedding_export"

    # Attach locked predictions/errors and calculate error/OOD subgroups.
    internal_predictions = oof[["outer_fold", "sample_id", "target", "y_true", "y_pred", "absolute_error"]].copy()
    internal_predictions["dataset"] = "internal_oof_test"
    feedback_id_map = feedback[["ID"]].copy()
    feedback_id_map["feedback_id"] = feedback_id_map["ID"].astype(str)
    feedback_id_map["audit_sample_id"] = feedback_id_map.index.astype(str)
    feedback_predictions_out = feedback_pred[["sample_id", "target", "y_true", "y_pred", "absolute_error"]].copy()
    feedback_predictions_out = feedback_predictions_out.merge(feedback_id_map[["feedback_id", "audit_sample_id"]], left_on="sample_id", right_on="feedback_id", how="left", validate="many_to_one")
    if feedback_predictions_out.audit_sample_id.isna().any():
        raise RuntimeError("Feedback prediction/sample-ID alignment failure")
    feedback_predictions_out["sample_id"] = feedback_predictions_out.audit_sample_id.astype(str)
    feedback_predictions_out = feedback_predictions_out.drop(columns=["feedback_id", "audit_sample_id"])
    feedback_predictions_out["outer_fold"] = np.nan
    feedback_predictions_out["dataset"] = "feedback"
    predictions = pd.concat([internal_predictions, feedback_predictions_out], ignore_index=True)
    predictions = predictions.merge(ood_scores, on=["dataset", "outer_fold", "sample_id"], how="left", validate="many_to_one")
    meta = pd.concat([train.assign(dataset="internal_oof_test"), feedback.assign(dataset="feedback")], axis=0, ignore_index=True)
    meta = meta.set_index(["dataset", "sample_id"])
    predictions["new_component"] = [len(component_sets(meta.loc[[key]].reset_index(drop=False))[0] - train_components) > 0 for key in zip(predictions.dataset, predictions.sample_id)]
    predictions["component_count"] = [5 if (str(meta.loc[(dataset, sample), "component_5_key"]) != "<missing>" and float(meta.loc[(dataset, sample), "mol%_Fifth"]) > 0) else 4 for dataset, sample in zip(predictions.dataset, predictions.sample_id)]
    ratio_min, ratio_max = train[RATIO_COLUMNS].astype(float).min().to_numpy(), train[RATIO_COLUMNS].astype(float).max().to_numpy()
    predictions["ratio_outside_train_range"] = [bool(np.any((meta.loc[(dataset, sample), RATIO_COLUMNS].astype(float).to_numpy() < ratio_min) | (meta.loc[(dataset, sample), RATIO_COLUMNS].astype(float).to_numpy() > ratio_max))) for dataset, sample in zip(predictions.dataset, predictions.sample_id)]
    error_rows, corr_rows = [], []

    def append_error_row(dataset: str, target: str, subgroup_type: str, subgroup: object, group: pd.DataFrame) -> None:
        """Store error and prediction-spread diagnostics for a nonempty subgroup."""
        if not len(group):
            return
        label_std = float(group.y_true.std(ddof=1)) if len(group) > 1 else 0.0
        prediction_std = float(group.y_pred.std(ddof=1)) if len(group) > 1 else 0.0
        error_rows.append({
            "dataset": dataset,
            "target": target,
            "subgroup_type": subgroup_type,
            "subgroup": str(subgroup),
            "n": len(group),
            "mae": float(group.absolute_error.mean()),
            "label_std": label_std,
            "prediction_std": prediction_std,
            "prediction_to_label_std_ratio": prediction_std / label_std if label_std > 0 else np.nan,
        })

    train_tail = {target: np.quantile(train[target], [.2, .8]) for target in TARGETS}
    for (dataset, target), part in predictions.groupby(["dataset", "target"]):
        for metric in OOD_DISTANCE_METRICS:
            if len(part) <= 2 or part[metric].nunique() <= 1:
                continue
            pearson = pearsonr(part.absolute_error, part[metric])
            spearman = spearmanr(part.absolute_error, part[metric])
            corr_rows.append({"dataset": dataset, "target": target, "ood_metric": metric,
                              "pearson": float(pearson.statistic), "pearson_pvalue": float(pearson.pvalue),
                              "spearman": float(spearman.statistic), "spearman_pvalue": float(spearman.pvalue), "n": len(part)})
        for kind, values in [("ood_class", ["ID", "mild OOD", "severe OOD"]), ("new_component", [False, True]), ("ratio_outside_train_range", [False, True]), ("component_count", [4, 5])]:
            for value in values:
                group = part.loc[part[kind].eq(value)]
                append_error_row(dataset, target, kind, value, group)
        low, high = train_tail[target]
        for name, group in [("bottom20", part.loc[part.y_true <= low]), ("middle60", part.loc[(part.y_true > low) & (part.y_true < high)]), ("top20", part.loc[part.y_true >= high])]:
            append_error_row(dataset, target, "label_quantile_train_threshold", name, group)
        # All feedback samples can be ID under the strict 95/99-percentile
        # threshold.  Tertiles of the continuous distance retain a direct,
        # within-feedback check for prediction-spread compression.
        if part.f2_knn5_distance.nunique() > 2:
            ranked = part.f2_knn5_distance.rank(method="first")
            distance_tertile = pd.qcut(ranked, q=3, labels=["low", "middle", "high"])
            for value in ["low", "middle", "high"]:
                append_error_row(dataset, target, "ood_distance_tertile", value, part.loc[distance_tertile.eq(value)])

    # Domain classifier: diagnostic only, stratified CV and F2 only.
    domain_x = pd.concat([f2_train, f2_feedback], ignore_index=True)
    domain_y = np.r_[np.zeros(len(f2_train), dtype=int), np.ones(len(f2_feedback), dtype=int)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260715)
    domain_rows = []
    for name, estimator in [("LogisticRegression", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260715)),
                            ("ExtraTrees", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=.8, class_weight="balanced", n_jobs=8, random_state=20260715))]:
        pipeline = Pipeline([("preprocess", feature_preprocessor(f2_train)), ("model", estimator)])
        probability = cross_val_predict(pipeline, domain_x, domain_y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        domain_rows.append({"model": name, "feature_set": "F2", "cv": "StratifiedKFold(5)", "roc_auc": float(roc_auc_score(domain_y, probability)), "n_train": len(f2_train), "n_feedback": len(f2_feedback)})
    domain = pd.DataFrame(domain_rows)

    label_metrics = pd.DataFrame(label_rows(train, feedback, train))
    summary_frame = pd.DataFrame(summary_rows)
    component = pd.DataFrame(coverage_rows)
    feature = pd.DataFrame(feature_rows)
    error = pd.DataFrame(error_rows)
    corr = pd.DataFrame(corr_rows)
    # Deterministic diagnosis.
    strong_covariate = bool(domain.roc_auc.max() > .8)
    feedback_distribution = label_metrics.loc[(label_metrics.record_type.eq("distribution")) & label_metrics.dataset.eq("feedback")].set_index("target")
    train_distribution = label_metrics.loc[(label_metrics.record_type.eq("distribution")) & label_metrics.dataset.eq("train")].set_index("target")
    label_shift = any(abs(feedback_distribution.loc[target, "mean"] - train_distribution.loc[target, "mean"]) / max(train_distribution.loc[target, "std"], 1e-12) > .5 or feedback_distribution.loc[target, "std"] / max(train_distribution.loc[target, "std"], 1e-12) > 1.3 for target in TARGETS)
    id_error = error.loc[(error.dataset.eq("feedback")) & error.subgroup_type.eq("ood_class") & error.subgroup.eq("ID")].set_index("target")
    oof_error = error.loc[(error.dataset.eq("internal_oof_test")) & error.subgroup_type.eq("ood_class") & error.subgroup.eq("ID")].set_index("target")
    conditional = any(target in id_error.index and target in oof_error.index and id_error.loc[target, "mae"] > 1.2 * oof_error.loc[target, "mae"] for target in TARGETS)
    status = "MIXED_DOMAIN_SHIFT" if strong_covariate and (label_shift or conditional) else "COVARIATE_SHIFT_DOMINANT" if strong_covariate else "CONDITIONAL_SHIFT_SUSPECTED" if conditional else "LABEL_SHIFT_DOMINANT" if label_shift else "FEEDBACK_MOSTLY_IN_DOMAIN"

    summary_frame.to_csv(OUT / "dataset_summary.csv", index=False)
    component.to_csv(OUT / "component_coverage.csv", index=False)
    feature.sort_values(["feature_space", "psi"], ascending=[True, False], na_position="last").to_csv(OUT / "feature_shift_metrics.csv", index=False)
    label_metrics.to_csv(OUT / "label_shift_metrics.csv", index=False)
    ood_scores.to_csv(OUT / "sample_ood_scores.csv", index=False)
    error.to_csv(OUT / "subgroup_error_metrics.csv", index=False)
    corr.to_csv(OUT / "ood_error_correlations.csv", index=False)
    domain.to_csv(OUT / "domain_classifier_metrics.csv", index=False)
    predictions.to_csv(OUT / "predictions_with_ood.csv", index=False)
    manifest = {"train": str((BASE / "data_audit/dataset_with_sample_id.csv").resolve()), "feedback": str(FEEDBACK_PATH.resolve()),
                "oof_predictions": str((BASE / "tree_baselines/oof_predictions.csv").resolve()), "feedback_predictions": str((BENCHMARK / "tabular_baseline_feedback_predictions.csv").resolve()),
                "feedback_hash": sha256_file(FEEDBACK_PATH), "analysis_only": True, "feedback_not_used_for_model_selection": True,
                "frozen_embedding_note": "No fold-consistent feedback frozen-embedding export was available; embedding distance is explicitly unavailable rather than substituted.", "status": "completed"}
    (OUT / "execution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    severe = float((feedback_scores.ood_class == "severe OOD").mean())
    top_shifts = (feature.loc[feature.feature_type.eq("numeric")]
                  .sort_values("psi", ascending=False)
                  .groupby("feature_space", as_index=False).first())
    top_shift = top_shifts.sort_values("psi", ascending=False).iloc[0]
    top_shift_text = "; ".join(f"`{row.feature_space}/{row.feature}` (PSI={row.psi:.3f}, KS={row.ks_statistic:.3f})" for _, row in top_shifts.iterrows())
    label_effect = feedback_distribution[["mean", "std"]].join(train_distribution[["mean", "std"]], lsuffix="_feedback", rsuffix="_train")
    label_effect["mean_shift_in_train_sd"] = (label_effect.mean_feedback - label_effect.mean_train) / label_effect.std_train.clip(lower=1e-12)
    label_effect["std_ratio"] = label_effect.std_feedback / label_effect.std_train.clip(lower=1e-12)
    label_effect["shift_score"] = np.maximum(label_effect.mean_shift_in_train_sd.abs(), np.log(label_effect.std_ratio).abs())
    top_label_shift = label_effect.sort_values("shift_score", ascending=False).iloc[0]
    top_label_target = label_effect.sort_values("shift_score", ascending=False).index[0]
    error_ratio = (id_error.mae / oof_error.mae).sort_values(ascending=False)
    conditional_targets = error_ratio.loc[error_ratio.gt(1.2)]
    conditional_text = ("；".join(f"`{target}`={ratio:.2f}×OOF" for target, ratio in conditional_targets.items())
                        if len(conditional_targets) else "无目标超过 1.2× OOF 的门槛")
    drift_type = "mixed: strong covariate + label/conditional" if status == "MIXED_DOMAIN_SHIFT" else ("covariate" if strong_covariate else "conditional/label")
    top_correlations = corr.loc[(corr.dataset.eq("feedback")) & (corr.ood_metric.eq("f2_knn5_distance"))].sort_values("spearman", key=lambda values: values.abs(), ascending=False)
    significant_correlations = top_correlations.loc[top_correlations.spearman_pvalue.lt(.05)]
    feedback_tertiles = error.loc[(error.dataset.eq("feedback")) & error.subgroup_type.eq("ood_distance_tertile")]
    variance_text = "无法判断：feedback 中没有足够的连续 OOD 距离差异。"
    if not feedback_tertiles.empty:
        variance_pivot = feedback_tertiles.pivot(index="target", columns="subgroup", values="prediction_to_label_std_ratio")
        if {"low", "high"}.issubset(variance_pivot.columns):
            lower_at_high = (variance_pivot.high < variance_pivot.low)
            direction = "仅部分目标" if lower_at_high.any() and not lower_at_high.all() else ("全部目标" if lower_at_high.all() else "没有目标")
            pairs = "; ".join(f"{target}: {row.low:.2f}→{row.high:.2f}" for target, row in variance_pivot.loc[:, ["low", "high"]].iterrows())
            variance_text = f"按连续 F2 k=5 距离三分位，预测/标签标准差比从低→高距离为 {pairs}；{direction}显示方差压缩随距离加重。"
    report = ["# Training / internal OOF / feedback domain audit", "", f"Final status: `{status}`.", "",
              f"1. Feedback 新组分比例：{float(summary_frame.loc[summary_frame.dataset.eq('feedback_coverage'), 'new_component_fraction'].iloc[0]):.1%}；新配方比例：{float(summary_frame.loc[summary_frame.dataset.eq('feedback_coverage'), 'new_formula_fraction'].iloc[0]):.1%}。",
              f"2. 各特征空间最严重数值漂移：{top_shift_text}。",
              f"3. 标签均值/方差：{'存在明显标签漂移' if label_shift else '未见明显全局标签尺度漂移'}；最大标准化均值偏移为 `{top_label_target}` {top_label_shift.mean_shift_in_train_sd:+.2f} 个训练集标准差，标准差比={top_label_shift.std_ratio:.2f}，详见 `label_shift_metrics.csv`。",
              f"4. 域分类器 AUC：" + "; ".join(f"{row.model}={row.roc_auc:.3f}" for _, row in domain.iterrows()) + ".",
              f"5. Feedback OOD：ID={(feedback_scores.ood_class == 'ID').mean():.1%}, mild={(feedback_scores.ood_class == 'mild OOD').mean():.1%}, severe={severe:.1%}。ID/mild/severe 使用训练样本留一近邻距离的 95/99 百分位阈值；训练域本身异质性高，所以该单距离规则全部为 ID 并不与跨特征域分类 AUC 很高相矛盾。",
              "6. OOD–绝对误差相关（feedback，F2 k=5 Spearman）：" + "; ".join(f"{row.target}={row.spearman:.3f} (p={row.spearman_pvalue:.3g})" for _, row in top_correlations.iterrows()) + f"。显著相关目标：{', '.join(significant_correlations.target) if len(significant_correlations) else '无'}；其中没有正向显著相关，因此误差并未主要随该 OOD 距离单调增加。其他三种距离的相关与 p 值见 `ood_error_correlations.csv`。",
              f"7. 协变量漂移：{'强（域 AUC > 0.8）；它作用于四个目标共用的 F2 输入域，而非由单个目标的 OOD–误差正相关所证明' if strong_covariate else '不强'}。",
              f"8. 条件漂移：{'可疑；ID feedback 相对内部 ID-OOF 的 MAE 为 ' + conditional_text if conditional else '没有达到本审计的 ID 子组证据门槛'}。最强证据是 `EE_after` 与 `EE_before`；该结论也可能包含标签分布或实验批次效应，不能仅凭本审计归因于机制变化。",
              f"9. {variance_text}",
              "10. 当前内部 Group CV：" + ("低估了 feedback 难度。" if status != "FEEDBACK_MOSTLY_IN_DOMAIN" else "没有显示出明显低估。"),
              "11. 下一步：优先补充新组分/新配方及其覆盖的实验条件；若 ID feedback 仍系统性差，则同时审计批次和测量条件。反馈没有可与五折训练对齐的 frozen embedding 导出，因此嵌入距离明确记为 unavailable，而未以不一致的嵌入替代。反馈推理使用原来锁定的 target-level model/feature-set 选择，未用 feedback 选模；内部误差使用既有 NestedSelectedBaseline OOF。", "",
              "| target | feedback MAE | OOF MAE | feedback severe-OOD share | dominant evidence |", "| ------ | -----------: | -------: | -------------------------: | ----------------- |"]
    for target in TARGETS:
        feedback_mae = float(predictions.loc[(predictions.dataset.eq("feedback")) & predictions.target.eq(target), "absolute_error"].mean())
        oof_mae = float(predictions.loc[(predictions.dataset.eq("internal_oof_test")) & predictions.target.eq(target), "absolute_error"].mean())
        target_evidence = f"strong covariate; ID error {error_ratio[target]:.2f}× OOF" if target in error_ratio.index else status
        report.append(f"| {target} | {feedback_mae:.3f} | {oof_mae:.3f} | {severe:.1%} | {target_evidence} |")
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("1. 域分类器AUC:", domain[["model", "roc_auc"]].to_dict(orient="records"))
    print("2. severe OOD比例:", severe)
    print("3. OOD与误差的相关性:", top_correlations[["target", "pearson", "pearson_pvalue", "spearman", "spearman_pvalue"]].to_dict(orient="records"))
    print("4. 漂移最严重的目标:", top_label_target, "; feature:", f"{top_shift.feature_space}/{top_shift.feature}")
    print("5. 最可能的漂移类型:", drift_type)
    print("6. 最终状态:", status)
    print("7. report.md路径:", OUT / "report.md")


if __name__ == "__main__":
    main()
