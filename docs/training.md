# Training and evaluation

Run subject-wise five-fold cross-validation:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/facename.json \
  --data data/facename.pt \
  --output outputs/facename \
  --device cuda:0
```

Choose an empty output directory. Use `--device cpu` for CPU execution or
`--splits path/to/splits.json` to reuse saved folds.

The FACENAME configuration searches all per-condition combinations of
`K = 5, 10, ..., 50` and three learning rates: 3e-4, 5e-4 and 1e-3.
This produces 3,000 candidates per fold. Add `--fixed-k` to use the configured
`k_per_condition` values while retaining the learning-rate search.

Training uses AdamW, batch size 64, weight decay 1e-4, gradient clipping at 1.0
and validation-loss early stopping for up to 80 epochs. Regression uses MSE.
For classification, set `num_classes` in the model configuration; the model
returns logits and training uses cross-entropy.

Each fold chooses K, learning rate and the checkpoint by validation loss.
Test targets are used only for evaluation after model selection.

| Output | Contents |
|:--|:--|
| `config.json`, `splits.json` | Run settings and sample indices |
| `selection_<fold>.json` | Candidate validation losses and selected epochs |
| `fold_<fold>.pt` | Selected model configuration, weights and hyperparameters |
| `predictions_<fold>.pt` | Test indices, predictions and targets |
| `summary.json` | Per-fold metrics and their mean and standard deviation |

Regression reports MAE, RMSE and R². Classification reports macro-AUC,
accuracy and per-class sensitivity. Macro-AUC is `null` for folds missing
one or more target classes.
