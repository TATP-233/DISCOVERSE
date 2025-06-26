# OpenPI
## 1. Environment Setup

```bash
conda create -n pi python=3.11.0
conda activate pi
cd DISCOVERSE
pip install -r requirements.txt
pip install -e .
cd policies/openpi/packages/openpi-client/
pip install -e .
cd ../..
pip install -e .
cd ../../submodules/lerobot
pip install -e .
```

## 2. Generate Data

### 2.1. raw to hdf5
```bash
cd DISCOVERSE
python3 policies/act/data_process/raw_to_hdf5.py -md mujoco -dir data -tn ${task_name}  -vn ${video_names}
# for example:
python3 policies/act/data_process/raw_to_hdf5.py -md mujoco -dir data -tn block_place -vn cam_0 cam_1
```

### 2.2. move data

```bash
mv data/hdf5/${task_name} policies/openpi/training_data
cd policies/openpi/training_data/
scp instructions/${task_name}.json ${task_name}/instructions.json
cd ..
```
After generating the HDF5 data, we can directly generate the LerobotDataset format data for OpenPI.
If you want to create a multi-task dataset, please place the corresponding task folders according to the example below.

```
training_data/  
├── task_1
|   ├── instructions.json  
|   ├── episode_0.hdf5  
|   ├── episode_1.hdf5  
|   ├── ...  
|
├── task_2
|   ├── instructions.json  
|   ├── episode_0.hdf5  
|   ├── episode_1.hdf5  
|   ├── ...  
├──...
```

### 2.3. modify config
- In `policies/openpi/src/openpi/training/config.py`, there is a dictionary called `_CONFIGS`. You can modify two pre-configured PI0 configurations I’ve written:
`pi0_base_aloha_robotwin_lora` 
`pi0_fast_aloha_robotwin_lora`
`pi0_base_aloha_robotwin_full`
`pi0_fast_aloha_robotwin_full`

- If your do not have enough gpu memory, you can set `fsdp_devices`, refer to `policies/openpi/src/openpi/training/config.py`.

- When you need to switch the robot or change its observations and actions, you can modify the `RepackTransform` in `_CONFIGS` within `policies/openpi/src/openpi/training/config.py`.

### 2.4. set cache dict
If you don't have enough disk space under the `~/.cache` path, please use the following command to set a different cache directory with sufficient space:
```bash
export HF_LEROBOT_HOME=/path/to/your/cache
# for example: 
mkdir -p ~/openpi_cache
export HF_LEROBOT_HOME=~/openpi_cache
```

### 2.5. generate training data
```bash
bash generate.sh ./training_data training_data
```

### 2.6. compute norm stats
```bash
python3 scripts/compute_norm_stats.py --config-name ${train_config_name}
# for example:
python3 scripts/compute_norm_stats.py --config-name pi0_base_aloha_full
```

## 3. finetune

```bash
export HF_LEROBOT_HOME=/path/to/your/cache
# for example: 
export HF_LEROBOT_HOME=~/openpi_cache
```
```bash
bash finetune.sh ${train_config_name} ${model_name}
# for example:
bash finetune.sh pi0_base_aloha_full model_a
```

## 4. Inference
```bash
bash eval.sh ${task_name} ${train_config_name} ${model_name} ${checkpoint}
# for example:
bash eval.sh block_place pi0_base_aloha_full model_a 9999
```
