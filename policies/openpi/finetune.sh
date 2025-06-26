train_config_name=$1
model_name=$2

WANDB_MODE=disabled python3 scripts/train.py $train_config_name --exp-name=$model_name --overwrite