DEBUG=False

task_name=${1}
train_config_name=${2} 
model_name=${3}
checkpoint_num=${4}
gpu_id=${5:-0}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 eval.py $task_name $train_config_name $model_name $checkpoint_num