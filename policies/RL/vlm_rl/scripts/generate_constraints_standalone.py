#!/usr/bin/env python3
"""
Standalone constraint generation script - does not require DISCOVERSE SimNode.

Directly loads MJCF scene with MuJoCo and generates VLM constraints.

Usage:
    python scripts/generate_constraints_standalone.py \
        --config configs/tasks/put_bottle_into_pot.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import mujoco

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keypoint_proposer import KeypointProposer
from src.annotated_renderer import AnnotatedRenderer
from src.constraint_generator import ConstraintGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate VLM constraints (standalone)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to task config YAML file")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/<task_name>/constraints)")
    parser.add_argument("--vlm_model", type=str, default="gpt-4o",
                        help="VLM model to use")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force regeneration even if cache exists")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_name = config.get("task_name", "unknown_task")
    instruction = args.instruction or config.get("instruction", "")

    if not instruction:
        print("Error: No instruction provided. Use --instruction or set in config.")
        return 1

    # Get MJCF file path
    mjcf_path = config.get("mjcf_file_path", "")
    if not mjcf_path:
        print("Error: mjcf_file_path not specified in config")
        return 1

    # Make path relative to vlm_rl directory
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mjcf_full_path = os.path.join(module_dir, mjcf_path)

    if not os.path.exists(mjcf_full_path):
        print(f"Error: MJCF file not found: {mjcf_full_path}")
        return 1

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(module_dir, "outputs", task_name, "constraints")

    print("=" * 60)
    print("STANDALONE CONSTRAINT GENERATION")
    print("=" * 60)
    print(f"\nTask: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"MJCF: {mjcf_full_path}")
    print(f"Output: {output_dir}")

    # Load MuJoCo scene
    print("\n1. Loading MuJoCo scene...")
    try:
        mj_model = mujoco.MjModel.from_xml_path(mjcf_full_path)
        mj_data = mujoco.MjData(mj_model)
    except Exception as e:
        print(f"Error loading MJCF: {e}")
        return 1

    # Reset to home keyframe if available
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    print("   [OK] Scene loaded and reset to home position")

    # Initialize keypoint proposer
    print("\n2. Proposing keypoints...")
    object_bodies = config.get("object_bodies", [])
    points_per_object = config.get("points_per_object", 5)

    # Create custom renderer with free camera
    from src.annotated_renderer import AnnotatedRenderer
    renderer = AnnotatedRenderer(
        mj_model=mj_model,
        mj_data=mj_data,
        width=config.get("image_width", 640),
        height=config.get("image_height", 480),
        camera_name=None,  # Use free camera
    )

    # Override camera settings for better VLM view
    camera_config = config.get("camera_config", {})
    renderer.free_camera_lookat = camera_config.get("lookat", [0.4, 0.0, 0.45])
    renderer.free_camera_distance = camera_config.get("distance", 1.2)
    renderer.free_camera_azimuth = camera_config.get("azimuth", 180)
    renderer.free_camera_elevation = camera_config.get("elevation", -30)

    proposer = KeypointProposer(
        mj_model=mj_model,
        mj_data=mj_data,
        renderer=renderer,
        points_per_object=points_per_object,
        include_center=config.get("keypoint_include_center", True),
        depth_search_radius=config.get("keypoint_depth_search_radius", 6),
    )

    proposal = proposer.propose(object_bodies)
    keypoints_3d = proposal.keypoints_3d
    keypoints_2d = proposal.keypoints_2d

    print(f"   Generated {len(keypoints_3d)} keypoints from {len(proposal.object_names)} objects:")
    for obj_name, (start, end) in proposal.object_keypoint_ranges.items():
        print(f"     {obj_name}: keypoints {start}-{end-1}")

    # Initialize renderer with custom camera settings
    print("\n3. Rendering annotated image...")

    raw_image = renderer.render_rgb()
    annotated_image = renderer.render_with_keypoints_2d_by_object(
        keypoints_2d,
        proposal.object_keypoint_ranges,
    )

    # Generate keypoint description
    keypoint_desc = renderer.generate_keypoint_description(
        keypoints_3d,
        proposal.object_keypoint_ranges,
    )

    print("\n   Keypoint description:")
    for line in keypoint_desc.split('\n'):
        print(f"     {line}")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save raw and annotated images
    import cv2
    raw_path = os.path.join(output_dir, "raw_scene.jpg")
    cv2.imwrite(raw_path, cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
    image_path = os.path.join(output_dir, "annotated_scene.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"\n4. Saved raw image to: {raw_path}")
    print(f"   Saved annotated image to: {image_path}")

    # Save keypoint data
    keypoint_data = {
        "keypoints_3d": keypoints_3d.tolist(),
        "object_keypoint_ranges": proposal.object_keypoint_ranges,
        "object_names": proposal.object_names,
    }
    keypoints_path = os.path.join(output_dir, "keypoints.yaml")
    with open(keypoints_path, "w") as f:
        yaml.dump(keypoint_data, f)
    print(f"   Saved keypoints to: {keypoints_path}")

    # Generate constraints using VLM
    print("\n5. Generating constraints using VLM...")
    print(f"   Model: {args.vlm_model}")

    generator = ConstraintGenerator(
        model=args.vlm_model,
        cache_results=not args.no_cache,
    )

    result = generator.generate(
        image=annotated_image,
        raw_image=raw_image,
        instruction=instruction,
        keypoint_description=keypoint_desc,
        output_dir=output_dir,
    )

    # Print results
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nNumber of stages: {result.num_stages}")
    print(f"Grasp keypoints: {result.grasp_keypoints}")
    print(f"Release keypoints: {result.release_keypoints}")

    for stage in range(1, result.num_stages + 1):
        subgoal_count = len(result.subgoal_constraints.get(stage, []))
        path_count = len(result.path_constraints.get(stage, []))
        print(f"Stage {stage}: {subgoal_count} subgoal constraints, {path_count} path constraints")

    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Review constraints in {output_dir}/")
    print(f"  2. Run training:")
    print(f"     python scripts/train.py \\")
    print(f"         --config {args.config} \\")
    print(f"         --constraints_dir {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
