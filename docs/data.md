# Data preparation

Use preprocessed ROI time series with one integer condition label per TR.
For FACENAME, label ENC, DIST and REC as 0, 1 and 2. Negative labels exclude
unmodeled TRs from condition FCs; whole-session FC uses all TRs.

For each subject:

```python
from cohyfuse.data import prepare_connectomes

condition_fc, whole_fc = prepare_connectomes(time_series, labels, num_conditions=3)
```

`time_series` has shape `(T, V)` and `labels` has shape `(T,)`.
Each condition requires at least two observations. FC uses Pearson correlation,
Fisher z-transformation and a zero diagonal. Whole-session FC is stored in
strict upper-triangle order from `torch.triu_indices(V, V, offset=1)`.

Stack subjects and save a dictionary with `torch.save`:

| Key | Shape | Contents |
|:--|:--|:--|
| `A_qs` | `(N, Q, V, V)` | Condition FC matrices |
| `fc_vec` | `(N, V*(V-1)/2)` | Whole-session FC vectors |
| `target` | `(N,)` | Regression values or integer class labels |
| `subject_id` | List of length `N` | Subject identifiers |

All numeric inputs must be finite. The FACENAME configuration expects
`Q=3` and `V=100`. Use the same condition and ROI order for every subject.
Classification labels range from 0 to `num_classes - 1`.

For saved folds, provide a JSON list of five dictionaries containing `train`,
`val` and `test` index lists. Indices refer to the input dataset's subject order.
Each fold must partition all samples, subjects must not cross partitions,
and each sample must appear in exactly one outer test fold.
