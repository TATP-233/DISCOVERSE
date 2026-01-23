#!/usr/bin/env python3
"""
Generate constraints for a task using VLM.

This script:
1. Loads the MuJoCo scene
2. Proposes keypoints from object geometries
3. Renders annotated image
4. Calls VLM to generate constraint functions
5. Saves results to output directory
"""

import argparse
import os
import sys
import yaml
import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keypoint_proposer import KeypointProposer
from src.annotated_renderer import AnnotatedRenderer
from src.constraint_generator import ConstraintGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate VLM constraints for a task")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to task config YAML file")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/<task_name>)")
    parser.add_argument("--vlm_model", type=str, default="gpt-4o",
                        help="VLM model to use")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force regeneration even if cache exists")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering during setup")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_name = config.get("task_name", "unknown_task")
    instruction = args.instruction or config.get("instruction", "")

    if not instruction:
        print("Error: No instruction provided. Use --instruction or set in config.")
        return 1

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
            task_name,
            "constraints"
        )

    print(f"Task: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"Output directory: {output_dir}")

    # Import DISCOVERSE components
    task_module = config.get("task_module", "discoverse.examples.tasks_mmk2.kiwi_pick")
    print(f"Loading task module: {task_module}")

    try:
        exec(f"from {task_module} import SimNode, cfg", globals())
    except ImportError as e:
        print(f"Error importing task module: {e}")
        print("Make sure DISCOVERSE is installed and the task module exists.")
        return 1

    # Configure task
    cfg_updates = config.get("cfg_updates", {})
    for key, value in cfg_updates.items():
        setattr(cfg, key, value)

    cfg.headless = not args.render
    cfg.sync = True

    # Create task environment
    print("Initializing simulation...")
    task_base = SimNode(cfg)
    task_base.reset()

    # Apply domain randomization if available
    if hasattr(task_base, 'domain_randomization'):
        task_base.domain_randomization()

    # Get MuJoCo model and data
    mj_model = task_base.mj_model
    mj_data = task_base.mj_data

    # Initialize keypoint proposer
    print("Proposing keypoints...")
    object_bodies = config.get("object_bodies", [])
    points_per_object = config.get("points_per_object", 5)

    proposer = KeypointProposer(
        mj_model=mj_model,
        mj_data=mj_data,
        points_per_object=points_per_object,
    )

    proposal = proposer.propose(object_bodies)
    keypoints_3d = proposal.keypoints_3d

    print(f"Generated {len(keypoints_3d)} keypoints from {len(proposal.object_names)} objects")
    for obj_name, (start, end) in proposal.object_keypoint_ranges.items():
        print(f"  {obj_name}: keypoints {start}-{end-1}")

    # Initialize renderer
    print("Rendering annotated image...")
    camera_name = config.get("camera_name", None)

    renderer = AnnotatedRenderer(
        mj_model=mj_model,
        mj_data=mj_data,
        width=config.get("image_width", 640),
        height=config.get("image_height", 480),
        camera_name=camera_name,
    )

    # Render annotated image
    annotated_image = renderer.render_with_keypoints_by_object(
        keypoints_3d,
        proposal.object_keypoint_ranges,
    )

    # Generate keypoint description
    keypoint_desc = renderer.generate_keypoint_description(
        keypoints_3d,
        proposal.object_keypoint_ranges,
    )

    print("\nKeypoint description:")
    print(keypoint_desc)

    # Save annotated image
    os.makedirs(output_dir, exist_ok=True)
    import cv2
    cv2.imwrite(
        os.path.join(output_dir, "annotated_scene.jpg"),
        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    )
    print(f"Saved annotated image to {output_dir}/annotated_scene.jpg")

    # Save keypoint data
    keypoint_data = {
        "keypoints_3d": keypoints_3d.tolist(),
        "object_keypoint_ranges": proposal.object_keypoint_ranges,
        "object_names": proposal.object_names,
    }
    with open(os.path.join(output_dir, "keypoints.yaml"), "w") as f:
        yaml.dump(keypoint_data, f)

    # Generate constraints using VLM
    print("\nGenerating constraints using VLM...")
    generator = ConstraintGenerator(
        model=args.vlm_model,
        cache_results=not args.no_cache,
    )

    result = generator.generate(
        image=annotated_image,
        instruction=instruction,
        keypoint_description=keypoint_desc,
        output_dir=output_dir,
    )

    print(f"\nGenerated constraints:")
    print(f"  Number of stages: {result.num_stages}")
    print(f"  Grasp keypoints: {result.grasp_keypoints}")
    print(f"  Release keypoints: {result.release_keypoints}")

    for stage in range(1, result.num_stages + 1):
        subgoal_count = len(result.subgoal_constraints.get(stage, []))
        path_count = len(result.path_constraints.get(stage, []))
        print(f"  Stage {stage}: {subgoal_count} subgoal constraints, {path_count} path constraints")

    print(f"\nResults saved to: {output_dir}")
    print("You can now use this directory with train.py --constraints_dir")

    # Cleanup
    del task_base

    return 0


if __name__ == "__main__":
    sys.exit(main())
