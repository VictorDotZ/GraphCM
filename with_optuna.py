import gc
import torch

import optuna

from run import parse_args
from dataset import Dataset
from model import Model


def objective(trial):
    num_steps = trial.suggest_int("num_steps", 2000, 20000)
    batch_size = 2 ** trial.suggest_int("batch_size", 6, 10)
    hidden_size = 2 ** trial.suggest_int("hidden_size", 3, 7)
    embed_size = 2 ** trial.suggest_int("embed_size", 3, 7)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    args = parse_args()

    args.num_steps = num_steps
    args.batch_size = batch_size
    args.hidden_size = hidden_size
    args.embed_size = embed_size
    args.dropout_rate = dropout_rate

    dataset = Dataset(args)

    args.train = True
    model = Model(
        args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset
    )
    model.train(dataset)

    args.train = False
    args.valid = True

    sum_click_loss = 0.0

    for i in range(args.num_iter):
        valid_batches = dataset.gen_mini_batches(
            "valid", dataset.validset_size, shuffle=False
        )
        valid_click_loss, _, _ = model.evaluate(valid_batches, dataset)
        sum_click_loss += valid_click_loss

    del model
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    return sum_click_loss / args.num_iter


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        # pruner=pruner,
        storage="sqlite:///optuna_GraphCM.sqlite3",
        study_name="optuna_GraphCM",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials:", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
