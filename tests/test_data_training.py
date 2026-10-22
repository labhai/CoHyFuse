import json
from pathlib import Path

import numpy as np
import pytest
import torch
import train as training

from cohyfuse import CoHyFuse
from cohyfuse.data import (
    functional_connectivity, load_connectomes, load_subject_splits,
    prepare_connectomes, subject_splits,
)
from train import run_fold, search_candidates


def test_fc_pools_repeated_conditions_and_uses_all_trs_for_whole_fc():
    torch.manual_seed(3)
    series = torch.randn(12, 4)
    labels = torch.tensor([0, 0, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1])
    conditions, whole = prepare_connectomes(series, labels, 2)
    expected = torch.atanh(torch.corrcoef(series[labels == 0].T).clamp(-1 + 1e-6, 1 - 1e-6))
    expected.fill_diagonal_(0)
    torch.testing.assert_close(conditions[0], expected)
    upper = torch.triu_indices(4, 4, offset=1)
    full = functional_connectivity(series)
    torch.testing.assert_close(whole, full[upper[0], upper[1]])
    changed = series.clone()
    changed[labels < 0] *= -4
    changed_conditions, changed_whole = prepare_connectomes(changed, labels, 2)
    torch.testing.assert_close(changed_conditions, conditions)
    assert not torch.allclose(changed_whole, whole)


def test_subject_splits_prevent_leakage():
    subjects = np.repeat(np.arange(50), 2)
    splits = subject_splits(subjects, seed=7)
    tests = []
    for split in splits:
        groups = [set(subjects[split[key]]) for key in ("train", "val", "test")]
        assert groups[0].isdisjoint(groups[1])
        assert groups[0].isdisjoint(groups[2])
        assert groups[1].isdisjoint(groups[2])
        assert [len(split[key]) for key in ("train", "val", "test")] == [64, 16, 20]
        tests.extend(split["test"])
    assert sorted(tests) == list(range(len(subjects)))


@pytest.mark.parametrize("classes", [None, 3])
def test_training_selection_checkpoint_and_inference(tmp_path, classes, monkeypatch):
    torch.manual_seed(4)
    target = torch.arange(30).remainder(3) if classes else torch.randn(30) * 5 + 40
    payload = {"A_qs": torch.randn(30, 2, 8, 8), "fc_vec": torch.randn(30, 28),
               "target": target, "subject_id": [str(i) for i in range(30)]}
    path = tmp_path / "input.pt"
    torch.save(payload, path)
    payload = load_connectomes(path)
    split = subject_splits(payload["subject_id"])[0]
    config = {
        "model": {"num_rois": 8, "k_per_condition": [2, 3], "hidden_dim": 8,
                  "fc_hidden_dim": 16, "heads": 2, "dropout": 0, "num_classes": classes},
        "training": {"learning_rates": [0.0003, 0.001], "epochs": 2, "patience": 1,
                     "k_values": [2, 3], "batch_size": 8, "weight_decay": 0.0001,
                     "grad_clip": 1.0, "seed": 42},
    }
    output = tmp_path / "output"
    output.mkdir()
    fit = training.fit_candidate

    def check_targets(model, loader, validation, settings, lr, device):
        torch.testing.assert_close(loader.dataset.tensors[2], target[split["train"]])
        torch.testing.assert_close(validation[2], target[split["val"]])
        return fit(model, loader, validation, settings, lr, device)

    monkeypatch.setattr(training, "fit_candidate", check_targets)
    result = run_fold(payload, split, config, 0, torch.device("cpu"), output)
    assert np.isfinite(result["validation_loss"])
    assert result["learning_rate"] in config["training"]["learning_rates"]
    checkpoint = torch.load(output / "fold_0.pt", weights_only=True)
    assert checkpoint["model"]["k_per_condition"] == result["k_per_condition"]
    trials = json.loads((output / "selection_0.json").read_text())
    assert len(trials) == 8
    assert result["validation_loss"] == min(trial["validation_loss"] for trial in trials)
    model = CoHyFuse(**checkpoint["model"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    saved = torch.load(output / "predictions_0.pt", weights_only=True)
    with torch.no_grad():
        indices = saved["indices"]
        prediction = model(payload["A_qs"][indices], payload["fc_vec"][indices])
    torch.testing.assert_close(prediction, saved["prediction"])
    if classes is None:
        expected_mae = (prediction - target[split["test"]]).abs().mean().item()
        assert result["metrics"]["MAE"] == pytest.approx(expected_mae)


def test_condition_wise_search_grid():
    config_path = Path(__file__).parents[1] / "configs" / "facename.json"
    candidates = list(search_candidates(json.loads(config_path.read_text())))
    neighborhoods = {tuple(model["k_per_condition"]) for model, _ in candidates}
    assert len(candidates) == 3000
    assert len(neighborhoods) == 1000
    assert (50, 45, 5) in neighborhoods and (5, 50, 10) in neighborhoods
    assert {lr for _, lr in candidates} == {3e-4, 5e-4, 1e-3}


def test_test_targets_do_not_affect_model_selection(tmp_path):
    torch.manual_seed(5)
    payload = {"A_qs": torch.randn(30, 2, 8, 8), "fc_vec": torch.randn(30, 28),
               "target": torch.randn(30), "subject_id": [str(i) for i in range(30)]}
    split = subject_splits(payload["subject_id"])[0]
    config = {
        "model": {"num_rois": 8, "k_per_condition": [2, 3], "hidden_dim": 8,
                  "fc_hidden_dim": 16, "heads": 2, "dropout": 0},
        "training": {"learning_rates": [0.001], "k_values": [2, 3], "epochs": 1,
                     "patience": 1, "batch_size": 8, "weight_decay": 0.0001,
                     "grad_clip": 1.0, "seed": 42},
    }
    selected = []
    for changed in (False, True):
        data = {**payload, "target": payload["target"].clone()}
        if changed:
            data["target"][split["test"]] += 10000
        output = tmp_path / str(changed)
        output.mkdir()
        result = run_fold(data, split, config, 0, torch.device("cpu"), output)
        checkpoint = torch.load(output / "fold_0.pt", weights_only=True)
        selected.append((result, checkpoint))
    for key in ("k_per_condition", "learning_rate", "validation_loss", "best_epoch"):
        assert selected[0][0][key] == selected[1][0][key]
    for key, value in selected[0][1]["state_dict"].items():
        torch.testing.assert_close(value, selected[1][1]["state_dict"][key], rtol=0, atol=0)
    assert selected[0][0]["metrics"]["MAE"] != selected[1][0]["metrics"]["MAE"]


def test_saved_splits_round_trip(tmp_path):
    subjects = np.repeat(np.arange(20), 2)
    splits = subject_splits(subjects)
    path = tmp_path / "splits.json"
    path.write_text(json.dumps([
        {key: value.tolist() for key, value in split.items()} for split in splits
    ]))
    loaded = load_subject_splits(path, subjects)
    for original, restored in zip(splits, loaded):
        for key in original:
            np.testing.assert_array_equal(original[key], restored[key])


@pytest.mark.parametrize("invalid", ["subject_leakage", "repeated_test", "fractional_index"])
def test_saved_splits_reject_invalid_partitions(tmp_path, invalid):
    subjects = np.repeat(np.arange(20), 2)
    splits = subject_splits(subjects)
    rows = [{key: value.tolist() for key, value in split.items()} for split in splits]
    if invalid == "subject_leakage":
        rows[0]["train"][0], rows[0]["val"][0] = rows[0]["val"][0], rows[0]["train"][0]
    elif invalid == "repeated_test":
        rows = [rows[0]] * 5
    else:
        rows[0]["train"][0] = 0.5
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_subject_splits(path, subjects)
