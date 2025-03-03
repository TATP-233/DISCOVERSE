import os
import sys
from typing import Sequence, Mapping, Any, Union

import numpy as np
import torch
from PIL import Image
from discoverse.envs.simulator import SimulatorBase
import mediapy
from params_proto import ParamsProto, Proto


def read_frame(key, cap):
    ret, frame = cap.read()
    return key, frame


# 背景掩码生成
def background_mask(foreground_list):
    results_np = np.ones_like(np.array(foreground_list[0])) * 255
    foreground = np.any(np.array(foreground_list) > 127, axis=0)
    results_np[foreground] = 0
    return results_np


class SampleforDR:
    def __init__(self, objs, robot_parts, cam_ids, save_dir, fps, max_vis_dis=1.0):
        self.objs = objs
        self.cam_ids = cam_ids
        self.results = [[] for cam_id in self.cam_ids]
        self.save_dir = save_dir
        self.fps = fps
        self.robot_parts = robot_parts
        self.max_vis_dis = max_vis_dis

    def reset(self):
        self.results = [[] for cam_id in self.cam_ids]

    # sample videos for randomization
    def sampling(self, simnode: SimulatorBase):
        if not hasattr(self, "robot_geom_start") and self.robot_parts:
            self.robot_geom_start = 1e6
            self.robot_geom_end = 0
            for part in self.robot_parts:
                start = simnode.mj_model.body(part).geomadr
                end = (
                    simnode.mj_model.body(part).geomadr
                    + simnode.mj_model.body(part).geomnum
                )
                self.robot_geom_start = min(self.robot_geom_start, start)
                self.robot_geom_end = max(self.robot_geom_end, end)

        renderer_seg = simnode.renderer._segmentation_rendering
        renderer_depth = simnode.renderer._depth_rendering
        for cam_id in self.cam_ids:
            simnode.renderer.enable_segmentation_rendering()
            simnode.renderer.update_scene(
                simnode.mj_data, simnode.camera_names[cam_id], simnode.options
            )
            seg = simnode.renderer.render()
            geom_ids_ori = seg[:, :, 0]

            frames = {}
            for obj in self.objs:
                mask = np.zeros_like(geom_ids_ori, dtype=np.uint8)
                mask[
                    (simnode.mj_model.body(obj).geomadr <= geom_ids_ori)
                    & (
                        geom_ids_ori
                        < simnode.mj_model.body(obj).geomadr
                        + simnode.mj_model.body(obj).geomnum
                    )
                ] = 255
                frames[obj] = mask

            if self.robot_parts:
                mask = np.zeros_like(geom_ids_ori, dtype=np.uint8)
                mask[
                    (self.robot_geom_start <= geom_ids_ori)
                    & (geom_ids_ori < self.robot_geom_end)
                ] = 255
                frames["robot"] = mask

            frames["background"] = background_mask([frame for frame in frames.values()])
            frames["cam"] = simnode.obs["img"][cam_id]
            depth = simnode.obs["depth"][cam_id].squeeze()
            if self.max_vis_dis >= 1.0:
                depth = np.clip(depth / self.max_vis_dis, 0, 1)
            else:
                depth = np.clip(depth, 0, self.max_vis_dis)  # 此时不需要缩放
            frames["depth"] = (depth * 255).astype(np.uint8)

            self.results[cam_id].append(frames)

        simnode.renderer._segmentation_rendering = renderer_seg
        simnode.renderer._depth_rendering = renderer_depth

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path_new = self.create_new_folder()
        for cam_id in self.cam_ids:
            save_path = os.path.join(save_path_new, f"{cam_id}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            video_list = self.results[cam_id][0].keys()
            for video in video_list:
                mediapy.write_video(
                    os.path.join(save_path, f"{video}.mp4"),
                    [frames[video] for frames in self.results[cam_id]],
                    fps=self.fps,
                )

    def create_new_folder(self):

        os.makedirs(self.save_dir, exist_ok=True)
        existing_folders = [
            f
            for f in os.listdir(self.save_dir)
            if os.path.isdir(os.path.join(self.save_dir, f)) and f.isdigit()
        ]

        if existing_folders:
            max_num = max(int(f) for f in existing_folders)
            new_num = max_num + 1
        else:
            new_num = 0

        new_folder = f"{new_num:03d}"
        new_folder_path = os.path.join(self.save_dir, new_folder)

        os.makedirs(new_folder_path)

        return new_folder_path


# def DRmerge(image, rgb, RDpart, rgbpart):

#     import numpy as np
#     image_np = np.array(image)
#     rgb_np = np.array(rgb)

#     results_np = np.zeros_like(image_np)

#     for rd_mask in RDpart:
#         rd_mask_np = np.array(rd_mask)
#         rd_mask_np = rd_mask_np > 127
#         results_np[rd_mask_np] = image_np[rd_mask_np]

#     for rgb_mask in rgbpart:
#         rgb_mask_np = np.array(rgb_mask)
#         rgb_mask_np = rgb_mask_np > 127
#         results_np[rgb_mask_np] = rgb_np[rgb_mask_np]

#     results = PImage.fromarray(results_np)
#     results.format = "jpeg"

#     return results


def pick(d, *keys, strict=False):
    """Pick keys"""
    _d = {}
    for k in keys:
        if k in d:
            _d[k] = d[k]
        elif strict:
            raise KeyError(k)
    return _d


def center_crop(img, new_width, new_height):
    """
    Crops the given NumPy image array to the specified width and height centered around the middle of the image.

    Parameters:
    img (numpy.ndarray): The image to be cropped (assumed to be in HxWxC format).
    new_width (int): The desired width of the cropped image.
    new_height (int): The desired height of the cropped image.

    Returns:
    numpy.ndarray: The cropped image.
    """

    height, width = img.shape[:2]

    # Calculate the starting points (top-left corner) of the crop
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    # Perform the crop
    cropped_img = img[start_y : start_y + new_height, start_x : start_x + new_width]

    return cropped_img


def center_crop_pil(img, new_width, new_height):
    """
    Crops the given PIL Image to the specified width and height centered around the middle of the image.

    Parameters:
    img (PIL.Image.Image): The image to be cropped.
    new_width (int): The desired width of the cropped image.
    new_height (int): The desired height of the cropped image.

    Returns:
    PIL.Image.Image: The cropped image.
    """
    width, height = img.size

    # Calculate the top-left corner of the crop box
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Perform the crop
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_builtin_extra_nodes  # init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_builtin_extra_nodes()
    # init_custom_nodes()


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory

    if name.startswith("/"):
        return name

    if path is None:
        path = os.getcwd()

    path_name = os.path.join(path, name)

    if os.path.exists(path_name):
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


class ComfyUIArgs(ParamsProto, pre="comfy"):
    config_path: str = Proto(
        "extra_model_paths.yaml",
        env="COMFYUI_CONFIG_PATH",
        help="allow environment override, for cluster launches",
    )


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from submodules.ComfyUI.utils.extra_config import load_extra_path_config

    print(">>>>>", ComfyUIArgs.config_path)
    extra_model_paths = find_path(ComfyUIArgs.config_path)
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def load_local_image(image_path: str):
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to RGB
    img = img.convert("RGB")

    # Convert the image data to a numpy array and normalize it
    img_data = np.array(img).astype(np.float32) / 255.0

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.from_numpy(img_data)

    # Add an extra dimension to the tensor
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor
