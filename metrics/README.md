# Metrics

Exported scalars, run logs and hyperparameter search results for the reported
runs. Training needs a ROCm-capable AMD GPU; these files make the numbers
inspectable without rerunning it.

## Contents

| File | Description |
|---|---|
| `scalars.csv.gz` | tensorboardX scalars, 64216 points. Columns: `run, event_file, tag, step, wall_time, value` |
| `coursework/scalars.csv.gz` | earlier series of runs, 1081554 points, mostly optuna trials |
| `optuna_GraphCM.sqlite3` | optuna storage: 101 completed trials, 1 failed |
| `optuna_trials.csv` | the same trials as a flat table |
| `logs/*.txt` | run logs: full argument namespace, dataset sizes, timings |

Scalar tags: `train/lr`, `train/loss`, `valid/click_loss`, `valid/perplexity`,
`test/click_loss`, `test/perplexity`.

## Notes

`summary`, `summary_2` and `summary_3` hold copies of the same event files, with
two extra runs in `summary`, so duplicates have to be filtered out by
`event_file`.

`event_file` carries the unix start time of a run, which links it to the matching
file under `logs/` and through it to the launch arguments. Weighted and uniform
neighbour sampling are not distinguishable by command line — they differ in the
code and in the graph file — so that pairing has to be made by time.

optuna stores exponents rather than values: `batch_size`, `hidden_size` and
`embed_size` are `k` in `2^k`, see `with_optuna.py`. Best trial is #80:
`num_steps` 6419, `batch_size` 2^7 = 128, `hidden_size` 2^5 = 32, `embed_size`
2^7 = 128, `dropout_rate` 0.4544, objective (validation click loss) 0.19435.

Checkpoints and preprocessed data are not part of the repository.
