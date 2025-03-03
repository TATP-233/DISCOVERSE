import os
import json
import mediapy
import logging
import numpy as np
import yaml


def recoder(save_path, obs_lst, act_lst):
    os.mkdir(save_path)
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
            },
            "act": act_lst,
            "obj_pose": {},
        }
        for name in obs_lst[0]["obj_pose"].keys():
            obj["obj_pose"][name] = [
                tuple(map(list, o["obj_pose"][name])) for o in obs_lst
            ]
        json.dump(obj, fp)

    mediapy.write_video(
        os.path.join(save_path, "arm_video.mp4"),
        [o["img"][0] for o in obs_lst],
        fps=cfg.render_set["fps"],
    )
    mediapy.write_video(
        os.path.join(save_path, "global_video.mp4"),
        [o["img"][1] for o in obs_lst],
        fps=cfg.render_set["fps"],
    )


class ColorCodes:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S,%f"[:-3]  # Date format, microseconds trimmed

    COLORS = {
        logging.DEBUG: ColorCodes.GREEN,
        logging.INFO: ColorCodes.RESET,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, ColorCodes.RESET)
        record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"
        record.msg = f"{color}{record.msg}{ColorCodes.RESET}"
        return super().format(record)


def setup_global_logger(logger, file, level=logging.INFO):
    # color formatter
    formatter = ColorFormatter(ColorFormatter.FORMAT)

    # steaming handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)

    logger.handlers = [file_handler, handler]
    logger.setLevel(level)


############ IO ################
def load_json(json_file):
    data = None
    with open(json_file, "r") as fin:
        data = json.load(fin)

    return data


def load_txt(file):
    with open(file) as f:
        return f.read().strip()


def write_txt(file, content):
    with open(file, "w+") as f:
        f.write(content)


def dump_json(json_data, output_dir):
    with open(output_dir, "w") as fout:
        fout.write(json.dumps(json_data, indent=2))


def load_yaml(yaml_file):
    config = None
    with open(yaml_file, "r") as fin:
        config = yaml.safe_load(fin)

    return config


def save_npz(data, output_dir, compressed=False):
    if compressed:
        np.savez_compressed(output_dir, **data)
    else:
        np.savez(output_dir, **data)


def load_npz(data_file, allow_pickle=True):
    return np.load(data_file, allow_pickle=allow_pickle)


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


def print_config(config, depth=0):
    config_str = ""
    for k, v in config.items():
        if isinstance(v, dict):
            config_str += "{}* {}\n:".format("  " * depth, k)
            config_str += print_config(v, depth + 1)
        else:
            config_str += "{}* {}: {}\n".format("  " * depth, k, v)

    return config_str
