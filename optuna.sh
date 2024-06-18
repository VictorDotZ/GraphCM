#!/usr/bin/bash 

export TQDM_DISABLE=1

python -u with_optuna.py \
    --optim adam \
    --eval_freq 100 \
    --check_point 20000 \
    --dataset VK \
    --combine exp_mul \
    --gnn_neigh_sample 0 \
    --gnn_concat False \
    --inter_neigh_sample 0 \
    --learning_rate 0.001 \
    --lr_decay 0.5 \
    --weight_decay 1e-5 \
    --patience 6 \
    --model_dir /mnt/home/victor/MODELS/GraphCM/models/ \
    --result_dir ./GraphCM/results/ \
    --summary_dir ./GraphCM/summary/ \
    --log_dir ./GraphCM/log/