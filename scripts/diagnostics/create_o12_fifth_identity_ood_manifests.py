#!/usr/bin/env python3
"""Create audited 80/10/10 O12 splits with disjoint Fifth identities only.

Allocation uses only canonical fifth-component molecular identity, group row
counts, and the recorded single/double class counts.  Target values and every
external dataset are excluded from allocation; target columns are inspected
only after splitting to report label availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


TARGETS = [
    'EE_before', 'EE_after', 'Aerosolization_Efficiency',
    'mRNA_Recovery_Efficiency', 'Norm_before', 'Norm_after',
]
SPLITS = ('train', 'val', 'test')
TARGET_RATIOS = np.array([.8, .1, .1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_fifth(smiles: object) -> str:
    """Return the molecular identity used for grouping.

    The loader represents an absent fifth component as ``[Fr]``.  It is not a
    special split case: all such rows are one ordinary Fifth identity and must
    therefore be assigned to exactly one split.  A Fifth identity is likewise
    allowed to contain a mixture of ``single`` and ``double`` formulations;
    class never subdivides an identity.
    """
    if pd.isna(smiles) or str(smiles).strip() in {'', 'nan', '[Fr]'}:
        return '[Fr]'
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f'Invalid Fifth_SMILE value: {smiles!r}')
    return Chem.MolToSmiles(molecule, canonical=True)


def fifth_identity(row: pd.Series) -> str:
    """Treat a zero/absent Fifth exactly like the loader's ``[Fr]`` placeholder."""
    fifth_name = str(row.Fifth).strip().lower()
    if fifth_name in {'0', '0.0', 'nan', ''}:
        return '[Fr]'
    return canonical_fifth(row.Fifth_SMILE)


def group_inventory(data: pd.DataFrame) -> pd.DataFrame:
    identity = data.apply(fifth_identity, axis=1)
    rows = []
    for identity_key, group in data.assign(_fifth_identity=identity).groupby('_fifth_identity', sort=True):
        names = sorted(set(group.Fifth.fillna('[missing]').astype(str)))
        classes = sorted(set(group.Fifth_class.fillna('[missing]').astype(str)))
        if any(value not in {'single', 'double', '[missing]'} for value in classes):
            raise ValueError(f'Unexpected Fifth_class for {names}: {classes}')
        rows.append({
            'fifth_identity': identity_key,
            'Fifth_identities': '|'.join(names),
            'Fifth_class_values': '|'.join(classes),
            'rows': len(group),
            'single_rows': int(group.Fifth_class.eq('single').sum()),
            'double_rows': int(group.Fifth_class.eq('double').sum()),
        })
    return pd.DataFrame(rows).sort_values('fifth_identity').reset_index(drop=True)


def score_assignment(assignment: np.ndarray, group_rows: np.ndarray,
                     group_singles: np.ndarray, group_doubles: np.ndarray,
                     target_rows: np.ndarray, global_single_ratio: float,
                     minimum_unique_fifth_per_split: int) -> float:
    rows = np.bincount(assignment, weights=group_rows, minlength=3)
    singles = np.bincount(assignment, weights=group_singles, minlength=3)
    doubles = np.bincount(assignment, weights=group_doubles, minlength=3)
    if np.any(rows == 0) or np.any(singles == 0) or np.any(doubles == 0):
        return np.inf
    # Strongly prioritize sample proportions, while avoiding a split that is
    # overwhelmingly one Fifth_class. No target values are involved.
    row_error = np.square((rows - target_rows) / target_rows).sum()
    class_error = np.square(singles / rows - global_single_ratio).sum()
    group_counts = np.array([(assignment == index).sum() for index in range(3)], dtype=float)
    if np.any(group_counts < minimum_unique_fifth_per_split):
        return np.inf
    group_error = np.square((group_counts - group_counts.sum() * TARGET_RATIOS)
                            / (group_counts.sum() * TARGET_RATIOS)).sum()
    return float(10.0 * row_error + class_error + .05 * group_error)


def allocate_groups(groups: pd.DataFrame, seed: int, trials: int,
                    minimum_unique_fifth_per_split: int) -> np.ndarray:
    """Find a seed-reproducible, near-80/10/10 group assignment.

    Every candidate is generated only from Fifth-group row/class inventory.
    Keeping a large identity in train is permitted: it is the only way to
    maintain useful holdout sample counts when an identity itself exceeds 10%.
    """
    rng = np.random.default_rng(seed)
    group_rows = groups.rows.to_numpy(dtype=float)
    group_singles = groups.single_rows.to_numpy(dtype=float)
    group_doubles = groups.double_rows.to_numpy(dtype=float)
    target_rows = group_rows.sum() * TARGET_RATIOS
    global_single_ratio = group_singles.sum() / group_rows.sum()
    probabilities = TARGET_RATIOS
    fixed_train = np.flatnonzero(groups.fifth_identity.eq('[Fr]').to_numpy())
    if len(fixed_train) > 1:
        raise RuntimeError('The [Fr] placeholder must correspond to at most one Fifth identity.')
    best_assignment, best_score = None, np.inf
    for _ in range(trials):
        assignment = rng.choice(3, size=len(groups), p=probabilities)
        # An absent fifth component is not a meaningful OOD molecular holdout.
        # Keep its ordinary, internally consistent identity entirely in train.
        assignment[fixed_train] = 0
        # Random candidates occasionally leave one holdout empty. Repair this
        # using only group sizes, then score the fully group-disjoint split.
        for split_index in (1, 2):
            if not np.any(assignment == split_index):
                donors = np.setdiff1d(np.flatnonzero(assignment == 0), fixed_train,
                                       assume_unique=True)
                if not len(donors):
                    raise RuntimeError('Cannot repair an empty holdout without moving fixed [Fr] from train.')
                move = donors[np.argmin(group_rows[donors])]
                assignment[move] = split_index
        value = score_assignment(
            assignment, group_rows, group_singles, group_doubles,
            target_rows, global_single_ratio, minimum_unique_fifth_per_split)
        if value < best_score:
            best_assignment, best_score = assignment.copy(), value
    if best_assignment is None or not np.isfinite(best_score):
        raise RuntimeError(f'Unable to make a valid fifth-identity split for seed {seed}.')
    return best_assignment


def check_leakage(manifest: pd.DataFrame) -> None:
    identities = {
        split: set(manifest.loc[manifest.split.eq(split), 'fifth_identity'])
        for split in SPLITS
    }
    overlaps = {
        f'{left}_{right}': sorted(identities[left].intersection(identities[right]))
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test'))
    }
    if any(overlaps.values()):
        raise RuntimeError(f'Fifth identity leakage detected: {overlaps}')
    fr_splits = set(manifest.loc[manifest.fifth_identity.eq('[Fr]'), 'split'])
    if fr_splits and fr_splits != {'train'}:
        raise RuntimeError(f'[Fr] / Fifth=0 must be train-only, found in: {sorted(fr_splits)}')


def split_audit(data: pd.DataFrame, manifest: pd.DataFrame, seed: int,
                manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    joined = data.assign(original_row_index=np.arange(len(data))).merge(
        manifest[['original_row_index', 'split', 'fifth_identity']], on='original_row_index',
        how='left', validate='one_to_one')
    rows = []
    detail = {'seed': seed, 'manifest': str(manifest_path.resolve()), 'splits': {}}
    for split in SPLITS:
        frame = joined.loc[joined.split.eq(split)]
        valid_counts = {target: int(frame[target].notna().sum()) for target in TARGETS}
        class_counts = frame.Fifth_class.fillna('[missing]').value_counts().to_dict()
        identities = sorted(frame.fifth_identity.unique())
        names = sorted(frame.Fifth.fillna('[missing]').astype(str).unique())
        row = {
            'seed': seed, 'split': split, 'rows': len(frame),
            'unique_fifth_identities': len(identities),
            'unique_fifth_names': len(names),
            'single_rows': int(class_counts.get('single', 0)),
            'double_rows': int(class_counts.get('double', 0)),
            'fr_rows': int(frame.fifth_identity.eq('[Fr]').sum()),
            'single_fraction': float(class_counts.get('single', 0) / len(frame)),
            'double_fraction': float(class_counts.get('double', 0) / len(frame)),
            **{f'valid_{target}': count for target, count in valid_counts.items()},
        }
        rows.append(row)
        detail['splits'][split] = {
            **row,
            'Fifth_identities': identities,
            'Fifth_names': names,
            'row_indices': frame.original_row_index.astype(int).tolist(),
        }
    return pd.DataFrame(rows), detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(100, 110)))
    parser.add_argument('--trials-per-seed', type=int, default=50000)
    parser.add_argument('--minimum-unique-fifth-per-split', type=int, default=5,
                        help='Hard minimum number of canonical Fifth identities in every split.')
    args = parser.parse_args()
    if (len(args.seeds) != len(set(args.seeds)) or args.trials_per_seed <= 0
            or args.minimum_unique_fifth_per_split <= 0):
        raise ValueError('Seeds must be unique; trials and minimum unique Fifth must be positive.')

    input_csv, output = args.input_csv.resolve(), args.output_dir.resolve()
    data = pd.read_csv(input_csv, dtype={'ID': str})
    required = {'ID', 'Fifth', 'Fifth_SMILE', 'Fifth_class', *TARGETS}
    if missing := required.difference(data.columns):
        raise ValueError(f'Input CSV misses columns: {sorted(missing)}')
    if len(data) != 700 or data.ID.isna().any() or data.ID.duplicated().any():
        raise ValueError('O12 Fifth-OOD protocol requires exactly 700 rows with unique IDs.')
    groups = group_inventory(data)
    if len(groups) < 3 * args.minimum_unique_fifth_per_split:
        raise ValueError(
            'Not enough Fifth identities to satisfy the requested minimum in all splits: '
            f'{len(groups)} < {3 * args.minimum_unique_fifth_per_split}.')
    output.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output / 'fifth_identity_inventory.csv', index=False)
    identity_by_row = data.apply(fifth_identity, axis=1)
    audit_frames, inventory_rows, assignments = [], [], {}
    for seed in args.seeds:
        assignment = allocate_groups(
            groups, seed, args.trials_per_seed, args.minimum_unique_fifth_per_split)
        split_by_identity = dict(zip(groups.fifth_identity, np.asarray(SPLITS)[assignment]))
        manifest = pd.DataFrame({
            'sample_id': data.ID.astype(str),
            'split': identity_by_row.map(split_by_identity),
            'original_row_index': np.arange(len(data), dtype=int),
            'fifth_identity': identity_by_row,
            'Fifth': data.Fifth.fillna('[missing]').astype(str),
            'Fifth_class': data.Fifth_class.fillna('[missing]').astype(str),
        })
        manifest['split_order'] = manifest.groupby('split', sort=False).cumcount()
        if manifest.split.isna().any() or manifest.sample_id.duplicated().any():
            raise RuntimeError(f'Invalid manifest coverage for seed {seed}.')
        check_leakage(manifest)
        unique_counts = manifest.groupby('split').fifth_identity.nunique()
        if any(int(unique_counts.get(split, 0)) < args.minimum_unique_fifth_per_split
               for split in SPLITS):
            raise RuntimeError(f'Minimum unique Fifth constraint failed for seed {seed}.')
        manifest_path = output / f'fifth_identity_manifest_seed{seed}.csv'
        manifest.to_csv(manifest_path, index=False)
        audit, detail = split_audit(data, manifest, seed, manifest_path)
        identity_assignment = dict(zip(groups.fifth_identity, np.asarray(SPLITS)[assignment]))
        assignment_serialized = json.dumps(identity_assignment, sort_keys=True, ensure_ascii=False)
        assignment_hash = hashlib.sha256(assignment_serialized.encode('utf-8')).hexdigest()
        if assignment_hash in assignments:
            raise RuntimeError(
                f'Duplicate complete split assignment for seeds {assignments[assignment_hash]} and {seed}.')
        assignments[assignment_hash] = seed
        detail['identity_assignment_sha256'] = assignment_hash
        detail['minimum_unique_fifth_per_split_required'] = args.minimum_unique_fifth_per_split
        detail['minimum_unique_constraint_passed'] = True
        audit_frames.append(audit)
        (output / f'fifth_identity_split_audit_seed{seed}.json').write_text(
            json.dumps(detail, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        counts = audit.set_index('split')
        inventory_rows.append({
            'seed': seed,
            'manifest': str(manifest_path.resolve()),
            'manifest_sha256': sha256(manifest_path),
            **{f'{split}_rows': int(counts.loc[split, 'rows']) for split in SPLITS},
            **{f'{split}_unique_fifth_identities': int(counts.loc[split, 'unique_fifth_identities'])
               for split in SPLITS},
            'identity_assignment_sha256': assignment_hash,
            'minimum_unique_fifth_per_split_required': args.minimum_unique_fifth_per_split,
            'minimum_unique_constraint_passed': True,
            'leakage_check': 'passed',
        })
    pd.concat(audit_frames, ignore_index=True).to_csv(output / 'split_audit_all_seeds.csv', index=False)
    pd.DataFrame(inventory_rows).to_csv(output / 'fifth_identity_manifest_inventory.csv', index=False)
    uniqueness_rows = []
    manifests = {
        seed: pd.read_csv(output / f'fifth_identity_manifest_seed{seed}.csv')
        for seed in args.seeds
    }
    for index, left_seed in enumerate(args.seeds):
        left = manifests[left_seed]
        for right_seed in args.seeds[index + 1:]:
            right = manifests[right_seed]
            left_split = left.set_index('fifth_identity').split
            right_split = right.set_index('fifth_identity').split
            # Deduplicate row-level entries before comparing group assignments.
            left_split = left_split[~left_split.index.duplicated(keep='first')]
            right_split = right_split[~right_split.index.duplicated(keep='first')]
            identical = bool(left_split.sort_index().equals(right_split.sort_index()))
            if identical:
                raise RuntimeError(f'Split uniqueness failure: seeds {left_seed} and {right_seed} are identical.')
            for split in ('val', 'test'):
                left_groups = set(left_split[left_split.eq(split)].index)
                right_groups = set(right_split[right_split.eq(split)].index)
                union = left_groups.union(right_groups)
                uniqueness_rows.append({
                    'left_seed': left_seed, 'right_seed': right_seed, 'split': split,
                    'identical_full_assignment': False,
                    'shared_fifth_identities': len(left_groups.intersection(right_groups)),
                    'union_fifth_identities': len(union),
                    'jaccard_overlap': len(left_groups.intersection(right_groups)) / len(union),
                })
    pd.DataFrame(uniqueness_rows).to_csv(output / 'split_uniqueness_audit.csv', index=False)
    protocol = {
        'input_csv': str(input_csv), 'input_sha256': sha256(input_csv),
        'rows': 700, 'seeds': args.seeds, 'trials_per_seed': args.trials_per_seed,
        'grouping_feature': 'canonical RDKit SMILES from Fifth_SMILE (one molecular identity per group)',
        'allocation_features': ['Fifth identity group row counts', 'Fifth_class row counts'],
        'target_values_used_for_allocation': False,
        'target_label_availability': 'audited after split creation only; never constrained or optimized',
        'absent_fifth_policy': '[Fr] / Fifth=0 is one ordinary molecular identity and remains split-disjoint',
        'absent_fifth_split_policy': '[Fr] / Fifth=0 is fixed entirely in train and is never a val/test OOD holdout',
        'mixed_single_double_identity_policy': 'one identity remains one group even if it contains both classes',
        'minimum_unique_fifth_per_split': args.minimum_unique_fifth_per_split,
        'split_uniqueness_audit': (
            'all seed-pair full identity assignments are required to differ; val/test group-set '
            'Jaccard overlap is reported descriptively'),
        'external_dataset_used': False,
        'target_value_or_external_data_used_for_model_selection': False,
        'split_target_ratios': {'train': .8, 'val': .1, 'test': .1},
        'leakage_invariant': 'No canonical Fifth identity occurs in more than one split.',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(pd.DataFrame(inventory_rows).to_string(index=False))


if __name__ == '__main__':
    main()
