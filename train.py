import argparse
import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cohyfuse import CoHyFuse
from cohyfuse.data import load_connectomes, load_subject_splits, subject_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(prediction: torch.Tensor, target: torch.Tensor, num_classes: int | None) -> dict:
    if num_classes is None:
        error = prediction - target
        variance = (target - target.mean()).square().sum()
        return {
            "MAE": error.abs().mean().item(),
            "RMSE": error.square().mean().sqrt().item(),
            "R2": (1 - error.square().sum() / variance).item() if variance > 0 else None,
        }
    probabilities = prediction.softmax(dim=-1).numpy()
    predicted = probabilities.argmax(axis=1)
    target = target.numpy()
    auc = None
    if len(np.unique(target)) == num_classes:
        if num_classes == 2:
            auc = roc_auc_score(target, probabilities[:, 1])
        else:
            auc = roc_auc_score(target, probabilities, multi_class="ovr", average="macro")
    result = {"macro_AUC": auc, "accuracy": accuracy_score(target, predicted)}
    sensitivity = recall_score(target, predicted, labels=list(range(num_classes)),
                               average=None, zero_division=0)
    result.update({f"sensitivity_{i}": float(value) for i, value in enumerate(sensitivity)})
    return result


def fit_candidate(model, train_loader, validation, settings, lr, device):
    criterion = nn.MSELoss() if model.num_classes is None else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=settings["weight_decay"])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for condition_fc, whole_fc, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(condition_fc.to(device), whole_fc.to(device)), target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(validation[0].to(device), validation[1].to(device))
            val_loss = criterion(prediction, validation[2].to(device)).item()
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss.")
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    return best_loss, best_state, best_epoch


def search_candidates(config):
    model, settings = config["model"], config["training"]
    values = settings.get("k_values")
    if values is not None:
        if not values or any(
            type(k) is not int or not 1 <= k < model["num_rois"] for k in values
        ):
            raise ValueError("k_values must contain integers in [1, num_rois - 1].")
        neighborhoods = product(sorted(set(values)), repeat=len(model["k_per_condition"]))
    else:
        neighborhoods = [model["k_per_condition"]]
    for k_values in neighborhoods:
        for lr in settings["learning_rates"]:
            yield {**model, "k_per_condition": list(k_values)}, lr


def run_fold(payload, split, config, fold, device, output):
    model_config, settings = config["model"], config["training"]
    classes = model_config.get("num_classes")
    target = payload["target"].float() if classes is None else payload["target"].long()
    tensors = (payload["A_qs"], payload["fc_vec"], target)
    validation = tuple(t[split["val"]] for t in tensors)
    selected = None
    trials = []
    seed = settings["seed"] + fold
    for candidate, lr in search_candidates(config):
        set_seed(seed)
        model = CoHyFuse(**candidate).to(device)
        generator = torch.Generator().manual_seed(seed)
        training = TensorDataset(*(t[split["train"]] for t in tensors))
        loader = DataLoader(
            training, batch_size=settings["batch_size"], shuffle=True, generator=generator
        )
        loss, state, epoch = fit_candidate(model, loader, validation, settings, lr, device)
        trial = {"k_per_condition": candidate["k_per_condition"], "learning_rate": lr,
                 "validation_loss": loss, "best_epoch": epoch}
        trials.append(trial)
        print(f"fold={fold} K={trial['k_per_condition']} lr={lr:g} val_loss={loss:.6g}",
              flush=True)
        if selected is None or loss < selected["validation_loss"]:
            selected = {"model": candidate, "state_dict": state, **trial, "seed": seed}
    (output / f"selection_{fold}.json").write_text(json.dumps(trials, indent=2) + "\n")
    model = CoHyFuse(**selected["model"]).to(device)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    with torch.no_grad():
        test = split["test"]
        batches = np.array_split(test, max(1, int(np.ceil(len(test) / settings["batch_size"]))))
        prediction = torch.cat([
            model(payload["A_qs"][batch].to(device), payload["fc_vec"][batch].to(device)).cpu()
            for batch in batches
        ])
    torch.save(selected, output / f"fold_{fold}.pt")
    torch.save({"indices": torch.as_tensor(test), "prediction": prediction, "target": target[test]},
               output / f"predictions_{fold}.pt")
    selection = {k: selected[k] for k in ("k_per_condition", "learning_rate",
                                        "validation_loss", "best_epoch")}
    return {"fold": fold, **selection, "metrics": metrics(prediction, target[test], classes)}


def main():
    parser = argparse.ArgumentParser(description="Subject-wise CoHyFuse cross-validation.")
    parser.add_argument("--config", default="configs/facename.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", help="Saved subject-wise five-fold split JSON.")
    parser.add_argument(
        "--fixed-k", action="store_true", help="Use the configured K without K search."
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device not in ("cpu", "cuda:0"):
        parser.error("Use cpu or a single visible GPU at cuda:0.")
    config = json.loads(Path(args.config).read_text())
    model_config, settings = config["model"], config["training"]
    if args.fixed_k:
        settings.pop("k_values", None)
    CoHyFuse(**model_config)
    positive = ("batch_size", "epochs", "patience", "grad_clip")
    if any(settings[key] <= 0 for key in positive) or settings["weight_decay"] < 0:
        parser.error("Training sizes and gradient limit must be positive; weight_decay >= 0.")
    if not settings["learning_rates"] or any(lr <= 0 for lr in settings["learning_rates"]):
        parser.error("Provide positive learning rates.")
    try:
        next(search_candidates(config))
    except ValueError as error:
        parser.error(str(error))
    payload = load_connectomes(args.data)
    expected = (
        len(model_config["k_per_condition"]), model_config["num_rois"], model_config["num_rois"]
    )
    if payload["A_qs"].shape[1:] != expected:
        parser.error("Dataset dimensions do not match the model configuration.")
    classes = model_config.get("num_classes")
    if classes is not None:
        target = payload["target"]
        if (target != target.long()).any() or (target < 0).any() or (target >= classes).any():
            parser.error("Classification targets must be integer labels in [0, num_classes - 1].")
    splits = load_subject_splits(args.splits, payload["subject_id"]) if args.splits else (
        subject_splits(payload["subject_id"], settings["seed"])
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Choose an empty output directory to preserve previous runs.")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "splits.json").write_text(json.dumps([
        {key: value.tolist() for key, value in split.items()} for split in splits
    ], indent=2) + "\n")
    results = []
    for fold, split in enumerate(splits):
        result = run_fold(payload, split, config, fold, torch.device(args.device), output)
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {}
    for name in results[0]["metrics"]:
        values = [r["metrics"][name] for r in results if r["metrics"][name] is not None]
        summary[name] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)), "folds": len(values)
        } if values else None
    report = {"folds": results, "summary": summary}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
