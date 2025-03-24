# usage

## Docker
Server: discoverse/s2r2025_server:v2.0
```
/workspace/SIM2REAL-2025/s2r2025
python3 s2r_server.py --round_id 1 --random_seed 34
```
Client: discoverse/s2r2025_client:baseline_act_v0
```
python3 /workspace/DISCOVERSE/discoverse/examples/s2r2025/run.py
```

# training tutorial

## generate data
the command below only generates 5, but recommend generating 300 examples

```
python3 discoverse/examples/s2r2025/pick_box.py --data_set_size 5 --dim17
```
- data_set_size: num of examples
- dim17: generage 17 joints data

or generate data automatically with multithreading.

```
python3 scripts/os_run.py --robot_name mmk2 --task_name pick_box --track_num 5 --nw 5 --dim17
```
- track_num: num of examples
- nw: num of threads
- dim17: generage 17 joints data

## transfer raw data to hdf5
```
python3 policies/act/data_process/raw_to_hdf5.py -md mujoco -dir data -tn mmk2_pick_box -vn cam_0 cam_1 cam_2
```

## train
```
python3 policies/train.py act -tn mmk2_pick_box
```

## infer
```
python3 policies/infer.py act -tn mmk2_pick_box -mts 300 -ts <checkpoint> -rn discoverse/examples/s2r2025/pick_box.py
```