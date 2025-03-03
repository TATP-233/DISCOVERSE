import os
from discoverse import DISCOVERSE_ROOT_DIR
from concurrent.futures import ThreadPoolExecutor
import argparse

if __name__ == "__main__":

    py_dir = os.popen("which python3").read().strip()

    parser = argparse.ArgumentParser(
        description="Run tasks with specified parameters. \ne.g. python3 os_run.py --robot_name airbot_play --task_name kiwi_place --track_num 100 --nw 8"
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        required=True,
        choices=["airbot_play", "mmk2", "hand_arm"],
        help="Name of the robot",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task, see discoverse/examples/tasks_{robot_name}",
    )
    parser.add_argument("--track_num", type=int, default=100, help="Number of tracks")
    parser.add_argument(
        "--nw", type=int, required=True, default=8, help="Number of workers"
    )
    args = parser.parse_args()

    robot_name = args.robot_name
    task_name = args.task_name
    track_num = args.track_num
    nw = args.nw

    def do_something(i):
        n = track_num // nw
        py_path = os.path.join(
            DISCOVERSE_ROOT_DIR,
            "discoverse/examples",
            f"tasks_{robot_name}/{task_name}.py",
        )
        os.system(f"{py_dir} {py_path} --data_idx {i*n} --data_set_size {n} --auto")

    # 使用with语句创建线程池，它会在with块结束时自动关闭
    with ThreadPoolExecutor(max_workers=nw) as executor:
        executor.map(do_something, range(nw))
