import os
import argparse
import json
import numpy as np
import zarr
from tqdm import tqdm
import cv2

obs_list = []
obs_rgb_cam_id = []


def read_ids_and_keys(source_dir, first_folder, json_name):  # 获取id_list and obs_keys
    folder_path = os.path.join(source_dir, first_folder)

    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    json_file = os.path.join(folder_path, json_name)
    ids = []
    obs_keys = []
    for video_file in video_files:
        if "cam_" in video_file:
            id = int(video_file.split("cam_")[1].split(".")[0])
            ids.append(id)  # 添加到 ID 列表

    with open(json_file, "r") as f:
        data = json.load(f)
        if "obs" in data:
            obs_keys = list(data["obs"].keys())  # 返回 obs 中的所有键
        else:
            print("Warning: 'obs' key not found in JSON.")

    return ids, obs_keys


def combine_and_save_to_zarr(source_dir, output_zarr):

    folders = sorted(
        [
            f
            for f in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, f))
        ]
    )

    obs_rgb_cam_id, obs_list = read_ids_and_keys(
        source_dir, folders[0], "obs_action.json"
    )

    root = zarr.group(output_zarr)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    all_actions = []
    all_obs = {obs: [] for obs in obs_list}
    all_images = {id: [] for id in obs_rgb_cam_id}
    episode_ends = []  # 用于记录每个episode的结束位置
    current_step = 0  # 用于追踪当前步数

    # 读取所有文件夹中的数据
    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(source_dir, folder)

        # 读取 json
        json_file = os.path.join(folder_path, "obs_action.json")
        with open(json_file, "r") as f:
            data = json.load(f)
            for obs in obs_list:
                all_obs[obs].extend(data["obs"][obs])
            all_actions.extend(data["act"])

        # 读取images
        for id in obs_rgb_cam_id:
            video_path = os.path.join(folder_path, f"cam_{id}.mp4")
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_images[id].append(frame)
                cap.release()
            else:
                print(f"Warning: {video_path} does not exist.")
        # 更新 episode_ends
        current_step += len(data["act"])  # 使用动作长度来计算step
        episode_ends.append(current_step)

    # 转换为numpy
    for obs in obs_list:
        all_obs[obs] = np.array(all_obs[obs])
    for id in obs_rgb_cam_id:
        all_images[id] = np.array(all_images[id])
    actions = np.array(all_actions)
    episode_ends = np.array(episode_ends)

    # 创建压缩器
    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2, blocksize=0)  # 自动块大小

    # 保存数据到 data 组
    for obs in obs_list:
        data_group.create_dataset(
            obs,
            data=all_obs[obs],
            chunks=(161, all_obs[obs].shape[1]),
            compressor=compressor,
            dtype="<f4",  # float32
            fill_value=0.0,
            order="C",
        )
    for id in obs_rgb_cam_id:
        data_group.create_dataset(
            f"image{id}",
            data=all_images[id],
            chunks=(161, *all_images[id].shape[1:]),
            compressor=compressor,
            dtype="<u1",  # uint8
            fill_value=0,
            order="C",
        )

    data_group.create_dataset(
        "action",
        data=actions,
        chunks=(161, actions.shape[1]),
        compressor=compressor,
        dtype="<f4",
        fill_value=0.0,
        order="C",
    )

    # 保存元数据到 meta 组
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        chunks=(161,),
        compressor=compressor,
        dtype="<i8",
        fill_value=0.0,
        order="C",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some directories.")
    parser.add_argument("-dir", "--work_dir", type=str, default="data")
    parser.add_argument("-tn", "--task_name", type=str, default="block_place")

    args = parser.parse_args()

    input_path = os.path.join(args.work_dir, args.task_name)
    output_path = f"{args.work_dir}/zarr"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f"{args.task_name}.zarr")

    combine_and_save_to_zarr(input_path, output_file)
