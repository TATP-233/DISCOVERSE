DEBUG=False
task_name=${1}
model_name=${2}
checkpoint=${3}
gpu_id=${4:-0}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 eval.py $task_name $model_name $checkpoint                                                                                                                                        