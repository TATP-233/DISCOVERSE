from .airbot_task_base import AirbotPlayTaskBase, recoder_airbot_play
from .mmk2_task_base import MMK2TaskBase, recoder_mmk2

import shutil


def copypy2(source_py, target_py):
    shutil.copy2(source_py, target_py)

    with open(target_py, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_main_block = False
    for line in lines:
        if in_main_block:
            break
        elif line.strip().startswith("if __name__"):
            in_main_block = True
            continue
        else:
            new_lines.append(line)

    mjcf_index = None
    for i, line in enumerate(new_lines):
        if line.strip().startswith("cfg.mjcf_file_path"):
            mjcf_index = i
            break

    if mjcf_index is not None:
        new_mjcf_path = 'os.path.abspath(__file__).replace(".py", ".mjb")'
        new_lines[mjcf_index] = f"cfg.mjcf_file_path = {new_mjcf_path}\n"

    with open(target_py, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
