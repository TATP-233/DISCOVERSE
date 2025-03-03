import cv2
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from params_proto.hyper import Sweep

from FlowCompute.flowcompute import FlowCompute
from discoverse.randomain.utils import pick, read_frame
from discoverse.randomain.workflow import ImageGen

import os
from discoverse import DISCOVERSE_ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images")

    parser.add_argument(
        "--seed", type=int, default=26, help="Random seed for picking the prompt"
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or gpu")

    # basic params
    parser.add_argument(
        "--task_name", type=str, default="block_place", help="Task name"
    )
    parser.add_argument("--work_dir", type=str, default="005", help="Working directory")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID")
    parser.add_argument(
        "--fore_objs",
        type=str,
        nargs="+",
        default=["block_green", "bowl_pink", "robot"],
        help="List of foreground objects",
    )

    # controlnet
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument(
        "--num_steps", type=int, default=2, help="Number of steps for image generation"
    )
    parser.add_argument(
        "--denoising_strength", type=float, default=1.0, help="Denoising strength"
    )
    parser.add_argument(
        "--control_strength", type=float, default=0.8, help="Control strength"
    )
    parser.add_argument(
        "--grow_mask_amount", type=int, default=0, help="Grow mask amount"
    )
    parser.add_argument(
        "--fore_grow_mask_amount",
        type=int,
        default=0,
        help="Foreground grow mask amount",
    )
    parser.add_argument(
        "--background_strength", type=float, default=0.2, help="Background strength"
    )
    parser.add_argument(
        "--fore_strength", type=float, default=2.0, help="Foreground strength"
    )

    # flow compute
    parser.add_argument(
        "--flow_interval",
        type=int,
        default=3,
        help="Frame interval for flow computation, set 1 for no flow interval",
    )
    parser.add_argument(
        "--flow_method",
        type=str,
        default="raft",
        help="Method for flow computation (rgb or raft)",
    )
    parser.add_argument(
        "--raft_model_path",
        type=str,
        default="./models/flow/raft-small.pth",
        help="Path to RAFT model checkpoint",
    )
    parser.add_argument("--raft_small", default=True, help="Use small RAFT model")
    parser.add_argument(
        "--raft_mixed_precision", default=False, help="Use mixed precision for RAFT"
    )
    parser.add_argument(
        "--raft_alternate_corr",
        default=False,
        help="Use efficient correlation implementation for RAFT",
    )

    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    imagen = ImageGen()

    control_parameters = pick(
        vars(args),
        "width",
        "height",
        "num_steps",
        "denoising_strength",
        "control_strength",
        "grow_mask_amount",
        "fore_grow_mask_amount",
        "background_strength",
        "fore_strength",
    )

    input_keys = ["cam", "depth", "background"] + args.fore_objs
    work_dir = os.path.join(
        DISCOVERSE_ROOT_DIR,
        f"data/{args.task_name}/segment/{args.work_dir}/{args.cam_id}",
    )
    input_paths = {k: f"{work_dir}/{k}.mp4" for k in input_keys}
    output_path = f"{work_dir}/output/{args.num_steps}.mp4"

    prompt_path = os.path.join(
        DISCOVERSE_ROOT_DIR,
        f"discoverse/randomain/prompts/{args.task_name}/prompts.jsonl",
    )
    prompts = Sweep.read(prompt_path)

    # input
    caps = {}
    for key, path in input_paths.items():
        caps[key] = cv2.VideoCapture(path)
        if not caps[key].isOpened():
            raise ValueError(f"cannot open video: {path}")

    fps = caps["cam"].get(cv2.CAP_PROP_FPS)
    frame_width = int(caps["cam"].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps["cam"].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if os.path.exists(os.path.dirname(output_path)):
        shutil.rmtree(os.path.dirname(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 读取视频
    print("--------Reading Video--------")
    frames_list = []
    finish = False

    with ThreadPoolExecutor() as executor:
        while True:
            frames = {}
            futures = {
                executor.submit(read_frame, key, cap): key for key, cap in caps.items()
            }
            for future in futures:
                key, frame = future.result()
                if frame is None:
                    finish = True
                else:
                    frames[key] = frame

            if finish:
                break
            frames_list.append(frames)

    # 生成
    Flow = FlowCompute(
        method=args.flow_method,
        model=args.raft_model_path,
        small=args.raft_small,
        mixed_precision=args.raft_mixed_precision,
        alternate_corr=args.raft_alternate_corr,
        device=args.device,
    )
    print("--------Generating--------")
    gen_list = []
    for i, frames in enumerate(frames_list):
        if i % args.flow_interval == 0:
            prompt = prompts[random.randint(0, len(prompts) - 1)]
            prompt = pick(prompt, "background", "negative", *args.fore_objs)

            gen = imagen.generate(
                depth=frames["depth"],
                masks={k: frames[k] for k in (args.fore_objs + ["background"])},
                prompt=prompt,
                **control_parameters,
            )
        else:
            flow = Flow.compute(frames_list[i - 1]["cam"], frames["cam"])
            gen = Flow.warp_forward(gen_list[-1], flow)

        gen_list.append(gen)

    # 保存
    for gen in gen_list:
        out.write(gen)

    # 结束
    for key, cap in caps.items():
        cap.release()
        out.release()


# image_foreDR = DRmerge(image, rgb, [block_mask, bowl_mask, robot_mask], [background_mask])
# image_backDR = DRmerge(image, rgb, [background_mask], [block_mask, bowl_mask, robot_mask])

if __name__ == "__main__":
    args = parse_args()
    main(args)
