import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def functional_connectivity(time_series: torch.Tensor) -> torch.Tensor:
    """Fisher-z Pearson FC from (T,V) ROI time series."""
    if time_series.ndim != 2 or time_series.shape[0] < 2:
        raise ValueError("time_series must have shape (T,V) with T >= 2.")
    if not torch.isfinite(time_series).all():
        raise ValueError("time_series must be finite.")
    values = time_series.float()
    centered = values - values.mean(dim=0, keepdim=True)
    normalized = centered / centered.square().sum(dim=0).sqrt().clamp_min(1e-8)
    correlation = (normalized.T @ normalized).clamp(-1 + 1e-6, 1 - 1e-6)
    fc = torch.atanh(correlation)
    return fc.masked_fill(torch.eye(fc.shape[0], device=fc.device, dtype=torch.bool), 0)


def prepare_connectomes(
    time_series: torch.Tensor, labels: torch.Tensor, num_conditions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative labels are excluded from condition FCs."""
    if labels.ndim != 1 or len(labels) != len(time_series) or num_conditions < 1:
        raise ValueError("Provide one label per TR and a positive condition count.")
    if labels.dtype not in (torch.int32, torch.int64) or (labels >= num_conditions).any():
        raise ValueError("Condition labels must be integers below num_conditions.")
    conditions = torch.stack([
        functional_connectivity(time_series[labels == q]) for q in range(num_conditions)
    ])
    whole = functional_connectivity(time_series)
    upper = torch.triu_indices(whole.shape[0], whole.shape[1], offset=1, device=whole.device)
    return conditions, whole[upper[0], upper[1]]


def load_connectomes(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"A_qs", "fc_vec", "target", "subject_id"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(f"Dataset must contain {sorted(required)}.")
    conditions, whole, target = (torch.as_tensor(payload[k]) for k in ("A_qs", "fc_vec", "target"))
    if conditions.ndim != 4 or conditions.shape[2] != conditions.shape[3]:
        raise ValueError("A_qs must have shape (N,Q,V,V).")
    n, q, v, _ = conditions.shape
    if n < 5 or q < 1 or v < 2:
        raise ValueError("Dataset requires N >= 5, Q >= 1 and V >= 2.")
    if whole.shape != (n, v * (v - 1) // 2) or target.shape != (n,):
        raise ValueError("fc_vec or target has an incompatible shape.")
    subjects = np.asarray(payload["subject_id"])
    if subjects.shape != (n,):
        raise ValueError("subject_id must contain one ID per sample.")
    if not all(torch.isfinite(t).all() for t in (conditions, whole, target)):
        raise ValueError("Dataset tensors must be finite.")
    return {"A_qs": conditions.float(), "fc_vec": whole.float(),
            "target": target, "subject_id": subjects}


def subject_splits(subject_ids, seed: int = 42) -> list[dict]:
    """Five subject-wise train/validation/test splits."""
    groups = np.asarray(subject_ids)
    if groups.ndim != 1 or len(np.unique(groups)) < 5:
        raise ValueError("At least five distinct subjects are required.")
    indices = np.arange(len(groups))
    splits = []
    for fold, (development, test) in enumerate(GroupKFold(5).split(indices, groups=groups)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
        train, val = next(splitter.split(development, groups=groups[development]))
        splits.append({"train": development[train], "val": development[val], "test": test})
    return splits


def load_subject_splits(path: str | Path, subject_ids) -> list[dict]:
    groups = np.asarray(subject_ids)
    rows = json.loads(Path(path).read_text())
    if groups.ndim != 1 or not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Provide subject IDs and five saved folds.")
    splits = []
    for row in rows:
        split = {}
        for key in ("train", "val", "test"):
            indices = np.asarray(row[key])
            if (indices.ndim != 1 or indices.size == 0
                    or not np.issubdtype(indices.dtype, np.integer)
                    or (indices < 0).any() or (indices >= len(groups)).any()):
                raise ValueError(f"Invalid {key} indices in split file.")
            split[key] = indices.astype(np.int64)
        combined = np.concatenate(list(split.values()))
        if not np.array_equal(np.sort(combined), np.arange(len(groups))):
            raise ValueError("Each fold must partition all samples exactly once.")
        subject_sets = [set(groups[split[key]]) for key in ("train", "val", "test")]
        if any(subject_sets[i] & subject_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("A subject occurs in multiple partitions of a fold.")
        splits.append(split)
    test_indices = np.concatenate([split["test"] for split in splits])
    if not np.array_equal(np.sort(test_indices), np.arange(len(groups))):
        raise ValueError("Each sample must occur in exactly one outer test fold.")
    return splits
