#!/bin/bash
# Default: Not a dev run
dev_run=false

# Check if "--dev" flag is passed
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) dev_run=true ;;  # Set dev_run to true if --dev flag is present
    esac
    shift
done

seeds=(2021 2022 2023)
dataset="ETTm2"

pred_len=96
seq_len=768
dropout=0.1
lr=0.001
batch_size=64
d_model=128
n_heads=8
e_layers=1
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
        --model.architecture Transformixer  \
        --model.architecture.n_heads $n_heads \
        --model.architecture.e_layers $e_layers \
        --model.architecture.d_model $d_model \
        --model.architecture.dropout $dropout \
        --optimizer.lr $lr \
        --lr_scheduler.constant_gamma_epochs $constant_gamma_epochs \
        --lr_scheduler.gamma $gamma \
        --lr_scheduler.cosine_epochs $cosine_epochs \
        --lr_scheduler.warmup_epochs $warmup_epochs \
        --trainer.logger.name "${dataset}_transformixer_${pred_len}_$seed" \
        --trainer.logger.project transformixer \
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=192
seq_len=768
dropout=0.1
lr=0.001
batch_size=128
d_model=128
n_heads=8
e_layers=1
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
        --model.architecture Transformixer  \
        --model.architecture.n_heads $n_heads \
        --model.architecture.e_layers $e_layers \
        --model.architecture.d_model $d_model \
        --model.architecture.dropout $dropout \
        --optimizer.lr $lr \
        --lr_scheduler.constant_gamma_epochs $constant_gamma_epochs \
        --lr_scheduler.gamma $gamma \
        --lr_scheduler.cosine_epochs $cosine_epochs \
        --lr_scheduler.warmup_epochs $warmup_epochs \
        --trainer.logger.name "${dataset}_transformixer_${pred_len}_$seed" \
        --trainer.logger.project transformixer \
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=336
seq_len=768
dropout=0.25
lr=0.001
batch_size=32
d_model=256
n_heads=8
e_layers=1
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
        --model.architecture Transformixer  \
        --model.architecture.n_heads $n_heads \
        --model.architecture.e_layers $e_layers \
        --model.architecture.d_model $d_model \
        --model.architecture.dropout $dropout \
        --optimizer.lr $lr \
        --lr_scheduler.constant_gamma_epochs $constant_gamma_epochs \
        --lr_scheduler.gamma $gamma \
        --lr_scheduler.cosine_epochs $cosine_epochs \
        --lr_scheduler.warmup_epochs $warmup_epochs \
        --trainer.logger.name "${dataset}_transformixer_${pred_len}_$seed" \
        --trainer.logger.project transformixer \
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=720
seq_len=768
dropout=0.1
lr=0.001
batch_size=256
d_model=128
n_heads=8
e_layers=1
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
        --model.architecture Transformixer  \
        --model.architecture.n_heads $n_heads \
        --model.architecture.e_layers $e_layers \
        --model.architecture.d_model $d_model \
        --model.architecture.dropout $dropout \
        --optimizer.lr $lr \
        --lr_scheduler.constant_gamma_epochs $constant_gamma_epochs \
        --lr_scheduler.gamma $gamma \
        --lr_scheduler.cosine_epochs $cosine_epochs \
        --lr_scheduler.warmup_epochs $warmup_epochs \
        --trainer.logger.name "${dataset}_transformixer_${pred_len}_$seed" \
        --trainer.logger.project transformixer \
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done