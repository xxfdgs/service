import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


COMPONENT_COLUMNS = [
    'IL_SMILE',
    'HL_SMILE',
    'Chol_SMILE',
    'PEG_SMILE',
    'Fifth_SMILE',
]
RATIO_COLUMNS = [
    'mol%_IL',
    'mol%_HL',
    'mol%_Chol',
    'mol%_PEG',
    'mol%_Fifth',
]
TARGET_COLUMNS = [
    'EE_before',
    'EE_after',
    'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency',
]
PREDICTION_COLUMNS = [
    'pred_EE_before',
    'pred_EE_after',
    'pred_Aero_Efficiency',
    'pred_Recovery_Efficiency',
]
FINGERPRINT_SIZE = 256


def featurize(frame, fingerprint_generator):
    feature_rows = []
    for _, row in frame.iterrows():
        component_features = []
        for component_column, ratio_column in zip(COMPONENT_COLUMNS,
                                                  RATIO_COLUMNS):
            fingerprint = np.zeros(FINGERPRINT_SIZE, dtype=np.uint8)
            smiles = row[component_column]
            if pd.notna(smiles):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is not None:
                    DataStructs.ConvertToNumpyArray(
                        fingerprint_generator.GetFingerprint(molecule),
                        fingerprint,
                    )
            ratio = float(row[ratio_column]) / 100.0
            component_features.append(fingerprint * ratio)

        ratio_features = row[RATIO_COLUMNS].to_numpy(dtype=float) / 100.0
        feature_rows.append(np.concatenate(component_features + [ratio_features]))
    return np.asarray(feature_rows)


def create_model():
    return ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=8,
        max_features=0.7,
        random_state=0,
        n_jobs=-1,
    )


def evaluate_group_cv(features, targets, groups):
    prediction = np.zeros_like(targets)
    splitter = GroupKFold(n_splits=5)

    for train_index, test_index in splitter.split(features, targets, groups):
        model = create_model()
        model.fit(features[train_index], targets[train_index])
        prediction[test_index] = model.predict(features[test_index])

    metrics = []
    for index, target_name in enumerate(TARGET_COLUMNS):
        metrics.append({
            'property': target_name,
            'mae': mean_absolute_error(targets[:, index], prediction[:, index]),
            'r2': r2_score(targets[:, index], prediction[:, index]),
        })
    return pd.DataFrame(metrics), prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base-csv',
        default='datasets_lrx/raw/input/20260703_sum.csv',
    )
    parser.add_argument(
        '--feedback-csv',
        default='datasets_lrx/raw/feedback/20260703_validation.csv',
    )
    parser.add_argument('--predict-csv')
    parser.add_argument(
        '--model-out',
        default='results/fingerprint_base_model.joblib',
    )
    parser.add_argument(
        '--prediction-out',
        default='results/fingerprint_base_predictions.csv',
    )
    parser.add_argument('--evaluate-group-cv', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=FINGERPRINT_SIZE,
    )
    base_frame = pd.read_csv(args.base_csv)
    base_features = featurize(base_frame, fingerprint_generator)
    base_targets = base_frame[TARGET_COLUMNS].to_numpy()

    if args.evaluate_group_cv:
        metrics, _ = evaluate_group_cv(
            base_features,
            base_targets,
            base_frame['Fifth_SMILE'].fillna('missing'),
        )
        print(metrics.to_string(index=False, float_format='{:.4f}'.format))

    model = create_model()
    model.fit(base_features, base_targets)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    prediction_frame = pd.read_csv(args.predict_csv or args.feedback_csv)
    prediction = model.predict(featurize(prediction_frame, fingerprint_generator))
    output = prediction_frame[['ID']].copy() if 'ID' in prediction_frame else pd.DataFrame()
    output[PREDICTION_COLUMNS] = prediction
    Path(args.prediction_out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.prediction_out, index=False)


if __name__ == '__main__':
    main()
