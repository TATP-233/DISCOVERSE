#!/usr/bin/env python3
"""
Convert a dataset (like new-desk) to MuJoCo-compatible format.

Dataset structure expected:
    dataset_dir/
    ├── 3d_assets/
    │   ├── {object_name}.obj      # Object mesh files
    │   ├── {object_name}.ply      # Gaussian splatting (optional)
    │   └── bg_*.ply               # Background files (ignored for objects)
    ├── masks/                      # Segmentation masks (optional)
    ├── input_image.jpg            # Scene image (optional)
    └── *.npy                       # Camera parameters (optional)

Output structure:
    output_dir/
    ├── meshes/
    │   ├── {object_name}.obj
    │   └── ...
    └── mjcf/
        └── scene.xml
"""

import argparse
import os
import shutil
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from mesh2mjcf import convex_decompose_mesh
except Exception:
    convex_decompose_mesh = None


@dataclass
class ObjectInfo:
    """Information about a detected object."""
    name: str
    obj_path: str
    ply_path: Optional[str]
    size: np.ndarray  # [width, depth, height]
    center: np.ndarray  # [x, y, z]
    min_bound: np.ndarray
    max_bound: np.ndarray
    vertex_count: int
    convex_parts: int = 0
    keyframe_pos: Optional[np.ndarray] = None


def parse_obj_file(obj_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Parse OBJ file and extract bounding box information.

    Returns:
        size, center, min_bound, max_bound, vertex_count
    """
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                # Handle both "v x y z" and "v x y z r g b" formats
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No vertices found in {obj_path}")

    vertices = np.array(vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    size = max_bound - min_bound
    center = (max_bound + min_bound) / 2

    return size, center, min_bound, max_bound, len(vertices)


def scan_dataset(dataset_dir: str) -> List[ObjectInfo]:
    """
    Scan dataset directory and detect all objects.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        List of ObjectInfo for each detected object
    """
    assets_dir = os.path.join(dataset_dir, "3d_assets")
    if not os.path.exists(assets_dir):
        raise FileNotFoundError(f"3d_assets directory not found in {dataset_dir}")

    objects = []
    seen_names = set()

    # Find all OBJ files (excluding background files)
    for filename in os.listdir(assets_dir):
        if not filename.endswith('.obj'):
            continue
        if filename.startswith('bg_'):
            continue

        # Extract object name (remove _mesh_final suffix if present)
        name = filename[:-4]  # Remove .obj
        if name.endswith('_mesh_final'):
            name = name[:-11]

        if name in seen_names:
            continue
        seen_names.add(name)

        # Prefer the base .obj file, fall back to _mesh_final.obj
        base_obj = os.path.join(assets_dir, f"{name}.obj")
        final_obj = os.path.join(assets_dir, f"{name}_mesh_final.obj")

        if os.path.exists(base_obj):
            obj_path = base_obj
        elif os.path.exists(final_obj):
            obj_path = final_obj
        else:
            continue

        # Check for PLY file
        ply_path = None
        for ply_name in [f"{name}.ply", f"{name}_gs_final.ply"]:
            ply_full = os.path.join(assets_dir, ply_name)
            if os.path.exists(ply_full):
                ply_path = ply_full
                break

        # Parse OBJ to get dimensions
        try:
            size, center, min_bound, max_bound, vertex_count = parse_obj_file(obj_path)
        except Exception as e:
            print(f"Warning: Failed to parse {obj_path}: {e}")
            continue

        keyframe_pos = _load_keyframe_position(assets_dir, name)

        objects.append(ObjectInfo(
            name=name,
            obj_path=obj_path,
            ply_path=ply_path,
            size=size,
            center=center,
            min_bound=min_bound,
            max_bound=max_bound,
            vertex_count=vertex_count,
            keyframe_pos=keyframe_pos,
        ))

    return objects


def estimate_collision_geometry(obj: ObjectInfo) -> Tuple[str, Dict]:
    """
    Estimate the best collision geometry type for an object based on its dimensions.

    Returns:
        (geom_type, geom_params)
    """
    w, d, h = obj.size

    # Determine aspect ratios
    max_dim = max(w, d, h)
    min_dim = min(w, d, h)
    aspect = max_dim / (min_dim + 1e-6)

    # Heuristics for geometry type selection
    if aspect > 3:
        # Elongated object -> cylinder or capsule
        if h > w and h > d:
            # Vertical cylinder (like a bottle)
            radius = max(w, d) / 2
            half_height = h / 2
            return "cylinder", {"size": [radius, half_height], "axis": "z"}
        else:
            # Horizontal cylinder
            if w > d:
                radius = max(d, h) / 2
                half_height = w / 2
                return "cylinder", {"size": [radius, half_height], "axis": "x"}
            else:
                radius = max(w, h) / 2
                half_height = d / 2
                return "cylinder", {"size": [radius, half_height], "axis": "y"}
    elif aspect < 1.5 and abs(w - d) / max(w, d) < 0.3:
        # Roughly square base -> cylinder or box
        if h < min(w, d) * 0.5:
            # Flat disk-like (like a pot/pan)
            radius = max(w, d) / 2
            half_height = h / 2
            return "cylinder", {"size": [radius, half_height], "axis": "z"}
        else:
            # Cube-ish -> box
            return "box", {"size": [w/2, d/2, h/2]}
    else:
        # Default to box
        return "box", {"size": [w/2, d/2, h/2]}


def estimate_mass(obj: ObjectInfo) -> float:
    """Estimate mass based on volume (assuming average density)."""
    volume = obj.size[0] * obj.size[1] * obj.size[2]
    # Assume density of ~500 kg/m³ (lighter than water, like plastic/wood)
    density = 500
    mass = volume * density
    # Clamp to reasonable range
    return max(0.01, min(mass, 5.0))


def generate_color(index: int) -> Tuple[float, float, float]:
    """Generate a distinct color for object visualization."""
    colors = [
        (0.6, 0.75, 0.2),   # Green-yellow
        (0.4, 0.35, 0.3),   # Brown
        (0.9, 0.85, 0.3),   # Yellow
        (0.3, 0.5, 0.8),    # Blue
        (0.8, 0.3, 0.3),    # Red
        (0.5, 0.8, 0.5),    # Light green
        (0.7, 0.4, 0.7),    # Purple
        (0.9, 0.6, 0.3),    # Orange
    ]
    return colors[index % len(colors)]


_TABLE_CENTER = (0.0, 0.5)
_TABLE_TOP_HALF = 0.02
_TABLE_EDGE_MARGIN = 0.12
_OBJECT_CLEARANCE = 0.001
_ROBOT_YAW = math.pi / 2


def _layout_object_positions(
    objects: List[ObjectInfo],
    table_size: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Lay out objects across the table facing the robot."""
    if not objects:
        return []

    n_objects = len(objects)
    table_center_x, table_center_y = _TABLE_CENTER
    table_width, table_depth = table_size

    # Place objects near the table center along y.
    y_pos = table_center_y

    max_footprint = max(max(obj.size[0], obj.size[1]) for obj in objects)
    min_spacing = max(0.18, max_footprint + 0.06)
    available = table_width - 2 * _TABLE_EDGE_MARGIN

    if n_objects == 1:
        x_positions = [table_center_x]
    else:
        spacing = min(min_spacing, available / max(n_objects - 1, 1))
        start = table_center_x - spacing * (n_objects - 1) / 2
        x_positions = [start + i * spacing for i in range(n_objects)]

    x_min = table_center_x - table_width / 2 + _TABLE_EDGE_MARGIN
    x_max = table_center_x + table_width / 2 - _TABLE_EDGE_MARGIN
    x_positions = [min(max(x, x_min), x_max) for x in x_positions]

    return [(x, y_pos) for x in x_positions]


def _object_z(table_height: float, obj: ObjectInfo) -> float:
    """Place object on table surface with a small clearance."""
    return table_height + _OBJECT_CLEARANCE + obj.size[2] / 2 - obj.center[2]


def _extract_keyframe_pos(arr: np.ndarray) -> Optional[np.ndarray]:
    """Extract xyz from a keyframe npy."""
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            item = arr.item() if arr.size == 1 else None
            if isinstance(item, dict):
                for key in ("pos", "position", "xyz"):
                    if key in item:
                        pos = np.asarray(item[key], dtype=float).reshape(-1)
                        if pos.size >= 3:
                            return pos[:3]
        flat = np.asarray(arr, dtype=float).reshape(-1)
        if flat.size >= 3:
            return flat[:3]
    return None


def _load_keyframe_position(assets_dir: str, name: str) -> Optional[np.ndarray]:
    """Load object keyframe position if present."""
    keyframe_path = os.path.join(assets_dir, f"{name}_keyframe.npy")
    if not os.path.exists(keyframe_path):
        return None
    try:
        arr = np.load(keyframe_path, allow_pickle=True)
    except Exception as exc:
        print(f"Warning: Failed to load keyframe {keyframe_path}: {exc}")
        return None
    pos = _extract_keyframe_pos(arr)
    if pos is None:
        print(f"Warning: Unsupported keyframe format in {keyframe_path}")
        return None
    return pos.astype(float)


def _resolve_object_positions(
    objects: List[ObjectInfo],
    table_size: Tuple[float, float],
    table_height: float,
) -> List[Tuple[float, float, float]]:
    """Resolve object positions from keyframes if available, else layout."""
    if not objects:
        return []

    key_positions = [obj.keyframe_pos for obj in objects]
    if all(pos is not None for pos in key_positions):
        pos_arr = np.stack([pos for pos in key_positions], axis=0).astype(float)

        # Treat keyframe positions as object centers; shift to mesh origin.
        centers = np.stack([obj.center for obj in objects], axis=0).astype(float)
        pos_arr[:, :2] = pos_arr[:, :2] - centers[:, :2]

        # Align dataset XY center to the configured table center.
        center_xy = np.median(pos_arr[:, :2], axis=0)
        dx = _TABLE_CENTER[0] - center_xy[0]
        dy = _TABLE_CENTER[1] - center_xy[1]

        pos_arr[:, 0] += dx
        pos_arr[:, 1] += dy

        return [
            (pos_arr[i, 0], pos_arr[i, 1], _object_z(table_height, obj))
            for i, obj in enumerate(objects)
        ]

    if any(pos is not None for pos in key_positions):
        print("Warning: Incomplete keyframes found. Falling back to layout positions.")

    layout = _layout_object_positions(objects, table_size)
    return [
        (x_pos, y_pos, _object_z(table_height, obj))
        for (x_pos, y_pos), obj in zip(layout, objects)
    ]


def generate_mjcf(
    objects: List[ObjectInfo],
    output_dir: str,
    scene_name: str = "converted_scene",
    include_robot: bool = True,
    table_size: Tuple[float, float] = (1.0, 0.7),
    table_height: float = 0.75,
) -> str:
    """
    Generate MJCF scene file.

    Args:
        objects: List of ObjectInfo
        output_dir: Output directory
        scene_name: Name for the scene
        include_robot: Whether to include Franka robot
        table_size: (width, depth) of table in meters
        table_height: Height of table surface

    Returns:
        Path to generated MJCF file
    """
    mjcf_dir = os.path.join(output_dir, "mjcf")
    os.makedirs(mjcf_dir, exist_ok=True)

    # Start building MJCF content
    mjcf_lines = [
        f'<mujoco model="{scene_name}">',
        '  <!-- Auto-generated scene from dataset conversion -->',
        '',
        '  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>',
        '  <option timestep="0.002" integrator="implicitfast" noslip_iterations="1"/>',
        '',
        '  <visual>',
        '    <global offwidth="1920" offheight="1080"/>',
        '    <quality shadowsize="4096" offsamples="4"/>',
        '    <headlight ambient="0.4 0.4 0.4"/>',
        '  </visual>',
        '',
    ]

    # Resolve object positions (keyframes > layout)
    object_positions = _resolve_object_positions(objects, table_size, table_height)

    # Add defaults
    mjcf_lines.extend(_generate_defaults(include_robot))

    # Add assets
    mjcf_lines.extend(_generate_assets(objects, include_robot))

    # Add worldbody
    mjcf_lines.extend(_generate_worldbody(objects, include_robot, table_size, table_height, object_positions))

    # Add contact exclusions (for robot gripper)
    if include_robot:
        mjcf_lines.extend(_generate_contacts())

    # Add equality constraints (for robot gripper)
    if include_robot:
        mjcf_lines.extend(_generate_equality())

    # Add actuators
    if include_robot:
        mjcf_lines.extend(_generate_actuators())

    # Add sensors
    if include_robot:
        mjcf_lines.extend(_generate_sensors())

    # Add keyframe
    mjcf_lines.extend(_generate_keyframe(objects, include_robot, object_positions))

    mjcf_lines.append('</mujoco>')

    # Write to file
    mjcf_path = os.path.join(mjcf_dir, f"{scene_name}.xml")
    with open(mjcf_path, 'w') as f:
        f.write('\n'.join(mjcf_lines))

    return mjcf_path


def _generate_defaults(include_robot: bool) -> List[str]:
    """Generate default section."""
    lines = ['  <!-- ============== DEFAULTS ============== -->', '  <default>']

    if include_robot:
        lines.extend([
            '    <!-- Panda arm defaults -->',
            '    <default class="panda">',
            '      <material specular="0.5" shininess="0.25"/>',
            '      <joint armature="0.1" damping="40" axis="0 0 1" range="-2.8973 2.8973"/>',
            '      <default class="finger">',
            '        <joint axis="0 1 0" type="slide" range="0 0.04"/>',
            '      </default>',
            '      <default class="visual">',
            '        <geom type="mesh" contype="0" conaffinity="0" group="2"/>',
            '      </default>',
            '      <default class="collision">',
            '        <geom group="3" type="mesh" contype="0" conaffinity="0"/>',
            '      </default>',
            '    </default>',
            '',
            '    <!-- Robotiq 2F-85 gripper defaults -->',
            '    <default class="2f85">',
            '      <mesh scale="0.001 0.001 0.001"/>',
            '      <general biastype="affine"/>',
            '      <joint axis="0 0 1"/>',
            '      <default class="driver">',
            '        <joint range="0 0.9" armature="0.005" damping="0.1" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>',
            '      </default>',
            '      <default class="follower">',
            '        <joint range="-0.872664 0.9" armature="0.001" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>',
            '      </default>',
            '      <default class="spring_link">',
            '        <joint range="-0.29670597283 0.9" armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>',
            '      </default>',
            '      <default class="coupler">',
            '        <joint range="-1.57 0" armature="0.001" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>',
            '      </default>',
            '      <default class="visual_gripper">',
            '        <geom type="mesh" contype="0" conaffinity="0" group="2" material="black"/>',
            '      </default>',
            '      <default class="collision_gripper">',
            '        <geom group="3" type="mesh" contype="0" conaffinity="0"/>',
            '        <default class="pad_box2">',
            '          <geom group="3" mass="1e-6" type="box" pos="0.041258 0 0.12875" size="0.004 0.011 0.01875"',
            '           solimp="0.99 0.995 0.01" solref="0.01 1" friction="1 0.005 0.0001" rgba="0.0 0.45 0.45 1" conaffinity="3"/>',
            '        </default>',
            '      </default>',
            '    </default>',
        ])

    lines.extend([
        '',
        '    <!-- Object defaults -->',
        '    <default class="object">',
        '      <geom condim="4" solimp="2 1 0.01" solref="0.01 1" friction="0.8 0.005 0.0001"/>',
        '    </default>',
        '  </default>',
        '',
    ])

    return lines


def _generate_assets(objects: List[ObjectInfo], include_robot: bool) -> List[str]:
    """Generate asset section."""
    lines = ['  <!-- ============== ASSETS ============== -->', '  <asset>']

    # Ground texture
    lines.extend([
        '    <!-- Ground texture -->',
        '    <texture name="grid" type="2d" builtin="checker" rgb1=".9 .8 .7" rgb2=".4 .4 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>',
        '    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>',
        '',
    ])

    if include_robot:
        lines.extend([
            '    <!-- Panda materials -->',
            '    <material class="panda" name="white" rgba="1 1 1 1"/>',
            '    <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>',
            '    <material class="panda" name="black" rgba="0.25 0.25 0.25 1"/>',
            '    <material class="panda" name="green" rgba="0 1 0 1"/>',
            '    <material class="panda" name="light_blue" rgba="0.039216 0.541176 0.780392 1"/>',
            '    <material class="panda" name="metal" rgba="0.58 0.58 0.58 1"/>',
            '',
        ])

    # Object materials
    lines.append('    <!-- Object materials -->')
    for i, obj in enumerate(objects):
        r, g, b = generate_color(i)
        lines.append(f'    <material name="{obj.name}_material" rgba="{r} {g} {b} 1" specular="0.5" shininess="0.5"/>')
    lines.extend([
        '    <material name="desk_material" rgba="0.6 0.5 0.4 1" specular="0.3" shininess="0.3"/>',
        '',
    ])

    if include_robot:
        # Robot meshes (reference the franka and robotiq directories)
        lines.extend(_get_robot_mesh_assets())

    # Object meshes
    lines.append('    <!-- Object meshes -->')
    for obj in objects:
        lines.append(f'    <mesh name="{obj.name}_mesh" file="{obj.name}.obj"/>')
        if obj.convex_parts > 0:
            for i in range(obj.convex_parts):
                lines.append(f'    <mesh name="{obj.name}_part_{i}" file="{obj.name}_part_{i}.obj"/>')

    lines.extend(['  </asset>', ''])
    return lines


def _get_robot_mesh_assets() -> List[str]:
    """Get robot mesh asset definitions."""
    return [
        '    <!-- Panda collision meshes -->',
        '    <mesh name="link0_c" file="franka/link0.stl"/>',
        '    <mesh name="link1_c" file="franka/link1.stl"/>',
        '    <mesh name="link2_c" file="franka/link2.stl"/>',
        '    <mesh name="link3_c" file="franka/link3.stl"/>',
        '    <mesh name="link4_c" file="franka/link4.stl"/>',
        '    <mesh name="link5_c0" file="franka/link5_collision_0.obj"/>',
        '    <mesh name="link5_c1" file="franka/link5_collision_1.obj"/>',
        '    <mesh name="link5_c2" file="franka/link5_collision_2.obj"/>',
        '    <mesh name="link6_c" file="franka/link6.stl"/>',
        '    <mesh name="link7_c" file="franka/link7.stl"/>',
        '',
        '    <!-- Panda visual meshes -->',
        '    <mesh file="franka/link0_0.obj"/>',
        '    <mesh file="franka/link0_1.obj"/>',
        '    <mesh file="franka/link0_2.obj"/>',
        '    <mesh file="franka/link0_3.obj"/>',
        '    <mesh file="franka/link0_4.obj"/>',
        '    <mesh file="franka/link0_5.obj"/>',
        '    <mesh file="franka/link0_7.obj"/>',
        '    <mesh file="franka/link0_8.obj"/>',
        '    <mesh file="franka/link0_9.obj"/>',
        '    <mesh file="franka/link0_10.obj"/>',
        '    <mesh file="franka/link0_11.obj"/>',
        '    <mesh file="franka/link1.obj"/>',
        '    <mesh file="franka/link2.obj"/>',
        '    <mesh file="franka/link3_0.obj"/>',
        '    <mesh file="franka/link3_1.obj"/>',
        '    <mesh file="franka/link3_2.obj"/>',
        '    <mesh file="franka/link3_3.obj"/>',
        '    <mesh file="franka/link4_0.obj"/>',
        '    <mesh file="franka/link4_1.obj"/>',
        '    <mesh file="franka/link4_2.obj"/>',
        '    <mesh file="franka/link4_3.obj"/>',
        '    <mesh file="franka/link5_0.obj"/>',
        '    <mesh file="franka/link5_1.obj"/>',
        '    <mesh file="franka/link5_2.obj"/>',
        '    <mesh file="franka/link6_0.obj"/>',
        '    <mesh file="franka/link6_1.obj"/>',
        '    <mesh file="franka/link6_2.obj"/>',
        '    <mesh file="franka/link6_3.obj"/>',
        '    <mesh file="franka/link6_4.obj"/>',
        '    <mesh file="franka/link6_5.obj"/>',
        '    <mesh file="franka/link6_6.obj"/>',
        '    <mesh file="franka/link6_7.obj"/>',
        '    <mesh file="franka/link6_8.obj"/>',
        '    <mesh file="franka/link6_9.obj"/>',
        '    <mesh file="franka/link6_10.obj"/>',
        '    <mesh file="franka/link6_11.obj"/>',
        '    <mesh file="franka/link6_12.obj"/>',
        '    <mesh file="franka/link6_13.obj"/>',
        '    <mesh file="franka/link6_14.obj"/>',
        '    <mesh file="franka/link6_15.obj"/>',
        '    <mesh file="franka/link6_16.obj"/>',
        '    <mesh file="franka/link7_0.obj"/>',
        '    <mesh file="franka/link7_1.obj"/>',
        '    <mesh file="franka/link7_2.obj"/>',
        '    <mesh file="franka/link7_3.obj"/>',
        '    <mesh file="franka/link7_4.obj"/>',
        '    <mesh file="franka/link7_5.obj"/>',
        '    <mesh file="franka/link7_6.obj"/>',
        '    <mesh file="franka/link7_7.obj"/>',
        '',
        '    <!-- Robotiq gripper meshes -->',
        '    <mesh name="robotiq_base" file="robotiq/base.stl"/>',
        '    <mesh file="robotiq/base_coupling.stl"/>',
        '    <mesh file="robotiq/c-a01-85-open.stl"/>',
        '    <mesh file="robotiq/driver.stl"/>',
        '    <mesh file="robotiq/coupler.stl"/>',
        '    <mesh file="robotiq/spring_link.stl"/>',
        '    <mesh file="robotiq/follower.stl"/>',
        '    <mesh file="robotiq/tongue.stl"/>',
        '',
    ]


def _generate_worldbody(
    objects: List[ObjectInfo],
    include_robot: bool,
    table_size: Tuple[float, float],
    table_height: float,
    object_positions: List[Tuple[float, float, float]],
) -> List[str]:
    """Generate worldbody section."""
    lines = ['  <!-- ============== WORLDBODY ============== -->', '  <worldbody>']

    # Lighting
    lines.extend([
        '    <!-- Lighting -->',
        '    <light directional="true" diffuse="0.3 0.3 0.3" pos="0.5 0.5 2.0"/>',
        '    <light directional="true" diffuse="0.2 0.2 0.2" pos="-0.5 0.5 2.0"/>',
        '',
        '    <!-- Ground plane -->',
        '    <geom name="ground" type="plane" pos="0 0 0" size="3 3 0.1" material="grid"',
        '          solimp=".9 .95 .001" solref="-10000 -1000"/>',
        '',
    ])

    # Table
    tw, td = table_size
    table_center_x, table_center_y = _TABLE_CENTER
    table_top_center_z = table_height - _TABLE_TOP_HALF
    table_top_thickness = _TABLE_TOP_HALF * 2
    leg_length = max(table_height - table_top_thickness, 0.05)
    leg_half = leg_length / 2
    leg_center_z = leg_half
    lines.extend([
        '    <!-- Desk/Table -->',
        f'    <body name="desk" pos="{table_center_x} {table_center_y} 0">',
        f'      <geom name="desk_top" type="box" size="{tw/2} {td/2} {_TABLE_TOP_HALF}" pos="0 0 {table_top_center_z}"',
        '            material="desk_material" class="object"/>',
        '      <!-- Table legs -->',
        f'      <geom type="cylinder" size="0.025 {leg_half}" pos="{tw/2-0.05} {td/2-0.05} {leg_center_z}" rgba="0.3 0.3 0.3 1"/>',
        f'      <geom type="cylinder" size="0.025 {leg_half}" pos="{tw/2-0.05} {-td/2+0.05} {leg_center_z}" rgba="0.3 0.3 0.3 1"/>',
        f'      <geom type="cylinder" size="0.025 {leg_half}" pos="{-tw/2+0.05} {td/2-0.05} {leg_center_z}" rgba="0.3 0.3 0.3 1"/>',
        f'      <geom type="cylinder" size="0.025 {leg_half}" pos="{-tw/2+0.05} {-td/2+0.05} {leg_center_z}" rgba="0.3 0.3 0.3 1"/>',
        '    </body>',
        '',
    ])

    # Robot
    if include_robot:
        lines.extend(_get_robot_body(table_height))

    # Objects - place them on the table
    lines.append('    <!-- ========== OBJECTS ========== -->')
    for i, obj in enumerate(objects):
        if obj.convex_parts <= 0:
            raise RuntimeError(f"Convex decomposition required but missing for {obj.name}.")
        x_pos, y_pos, z_pos = object_positions[i]

        mass = estimate_mass(obj)

        # Calculate inertia (rough approximation)
        w, d, h = obj.size
        ix = mass * (d*d + h*h) / 12
        iy = mass * (w*w + h*h) / 12
        iz = mass * (w*w + d*d) / 12
        diaginertia = f"{ix:.6f} {iy:.6f} {iz:.6f}"

        lines.extend([
            f'    <!-- {obj.name}: {obj.size[0]*100:.1f}cm x {obj.size[1]*100:.1f}cm x {obj.size[2]*100:.1f}cm -->',
            f'    <body name="{obj.name}" pos="{x_pos:.3f} {y_pos:.3f} {z_pos:.3f}">',
            '      <joint type="free" frictionloss="0.0001"/>',
            f'      <inertial pos="0 0 0" mass="{mass:.3f}" diaginertia="{diaginertia}"/>',
            '      <!-- Visual mesh -->',
            f'      <geom name="{obj.name}_visual" type="mesh" mesh="{obj.name}_mesh" material="{obj.name}_material"',
            '            contype="0" conaffinity="0" group="2"/>',
            '      <!-- Collision geometry -->',
        ])

        # Add collision geometry (convex parts only)
        for i in range(obj.convex_parts):
            lines.append(
                f'      <geom name="{obj.name}_collision_{i}" type="mesh" mesh="{obj.name}_part_{i}"'
                ' class="object" rgba="1 1 1 0"/>'
            )

        lines.extend([
            '    </body>',
            '',
        ])

    lines.append('  </worldbody>')
    lines.append('')
    return lines


def _get_robot_body(table_height: float) -> List[str]:
    """Get robot body definition (Franka + Robotiq)."""
    # This is a simplified version - the full robot body definition is quite long
    # We'll include it as an external file reference or inline
    robot_z = table_height / 2
    half = _ROBOT_YAW / 2
    qw = math.cos(half)
    qz = math.sin(half)

    return [
        '    <!-- ========== FRANKA PANDA + ROBOTIQ ========== -->',
        f'    <body name="link0" childclass="panda" gravcomp="1" pos="0 0 {robot_z:.3f}" quat="{qw:.6f} 0 0 {qz:.6f}">',
        '      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"',
        '        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>',
        '      <geom mesh="link0_0" material="off_white" class="visual"/>',
        '      <geom mesh="link0_1" material="black" class="visual"/>',
        '      <geom mesh="link0_2" material="off_white" class="visual"/>',
        '      <geom mesh="link0_3" material="black" class="visual"/>',
        '      <geom mesh="link0_4" material="off_white" class="visual"/>',
        '      <geom mesh="link0_5" material="black" class="visual"/>',
        '      <geom mesh="link0_7" material="white" class="visual"/>',
        '      <geom mesh="link0_8" material="white" class="visual"/>',
        '      <geom mesh="link0_9" material="black" class="visual"/>',
        '      <geom mesh="link0_10" material="off_white" class="visual"/>',
        '      <geom mesh="link0_11" material="white" class="visual"/>',
        '      <geom mesh="link0_c" class="collision"/>',
        '      <body name="link1" pos="0 0 0.333" gravcomp="1">',
        '        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"',
        '          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>',
        '        <joint name="joint1" range="-2.8973 2.8973"/>',
        '        <geom material="white" mesh="link1" class="visual"/>',
        '        <geom mesh="link1_c" class="collision"/>',
        '        <body name="link2" quat="1 -1 0 0" gravcomp="1">',
        '          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"',
        '            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>',
        '          <joint name="joint2" range="-1.7628 1.7628"/>',
        '          <geom material="white" mesh="link2" class="visual"/>',
        '          <geom mesh="link2_c" class="collision"/>',
        '          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0" gravcomp="1">',
        '            <joint name="joint3" range="-2.8973 2.8973"/>',
        '            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"',
        '              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>',
        '            <geom mesh="link3_0" material="white" class="visual"/>',
        '            <geom mesh="link3_1" material="white" class="visual"/>',
        '            <geom mesh="link3_2" material="white" class="visual"/>',
        '            <geom mesh="link3_3" material="black" class="visual"/>',
        '            <geom mesh="link3_c" class="collision"/>',
        '            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0" gravcomp="1">',
        '              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"',
        '                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>',
        '              <joint name="joint4" range="-3.0718 -0.0698"/>',
        '              <geom mesh="link4_0" material="white" class="visual"/>',
        '              <geom mesh="link4_1" material="white" class="visual"/>',
        '              <geom mesh="link4_2" material="black" class="visual"/>',
        '              <geom mesh="link4_3" material="white" class="visual"/>',
        '              <geom mesh="link4_c" class="collision"/>',
        '              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0" gravcomp="1">',
        '                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"',
        '                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>',
        '                <joint name="joint5" damping="2" range="-2.8973 2.8973"/>',
        '                <geom mesh="link5_0" material="black" class="visual"/>',
        '                <geom mesh="link5_1" material="white" class="visual"/>',
        '                <geom mesh="link5_2" material="white" class="visual"/>',
        '                <geom mesh="link5_c0" class="collision"/>',
        '                <geom mesh="link5_c1" class="collision"/>',
        '                <geom mesh="link5_c2" class="collision"/>',
        '                <body name="link6" quat="1 1 0 0" gravcomp="1">',
        '                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"',
        '                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>',
        '                  <joint name="joint6" range="-0.0175 3.7525" damping="2"/>',
        '                  <geom mesh="link6_0" material="off_white" class="visual"/>',
        '                  <geom mesh="link6_1" material="white" class="visual"/>',
        '                  <geom mesh="link6_2" material="black" class="visual"/>',
        '                  <geom mesh="link6_3" material="white" class="visual"/>',
        '                  <geom mesh="link6_4" material="white" class="visual"/>',
        '                  <geom mesh="link6_5" material="white" class="visual"/>',
        '                  <geom mesh="link6_6" material="white" class="visual"/>',
        '                  <geom mesh="link6_7" material="light_blue" class="visual"/>',
        '                  <geom mesh="link6_8" material="light_blue" class="visual"/>',
        '                  <geom mesh="link6_9" material="black" class="visual"/>',
        '                  <geom mesh="link6_10" material="black" class="visual"/>',
        '                  <geom mesh="link6_11" material="white" class="visual"/>',
        '                  <geom mesh="link6_12" material="green" class="visual"/>',
        '                  <geom mesh="link6_13" material="white" class="visual"/>',
        '                  <geom mesh="link6_14" material="black" class="visual"/>',
        '                  <geom mesh="link6_15" material="black" class="visual"/>',
        '                  <geom mesh="link6_16" material="white" class="visual"/>',
        '                  <geom mesh="link6_c" class="collision"/>',
        '                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0" gravcomp="1">',
        '                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"',
        '                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>',
        '                    <joint name="joint7" damping="2" range="-2.8973 2.8973"/>',
        '                    <geom mesh="link7_0" material="white" class="visual"/>',
        '                    <geom mesh="link7_1" material="black" class="visual"/>',
        '                    <geom mesh="link7_2" material="black" class="visual"/>',
        '                    <geom mesh="link7_3" material="black" class="visual"/>',
        '                    <geom mesh="link7_4" material="black" class="visual"/>',
        '                    <geom mesh="link7_5" material="black" class="visual"/>',
        '                    <geom mesh="link7_6" material="black" class="visual"/>',
        '                    <geom mesh="link7_7" material="white" class="visual"/>',
        '                    <geom mesh="link7_c" class="collision"/>',
        '                    <!-- Robotiq 2F-85 Gripper -->',
        '                    <body name="robotiq_base" childclass="2f85" quat="1 0 0 -1" pos="0 0 0.11" gravcomp="1">',
        '                      <inertial mass="0.777441" pos="0 -2.70394e-05 0.0354675" quat="1 -0.00152849 0 0"',
        '                        diaginertia="0.000260285 0.000225381 0.000152708"/>',
        '                      <geom class="visual_gripper" pos="0 0 0.0108" quat="0 0 0 1" mesh="robotiq_base"/>',
        '                      <geom class="visual_gripper" pos="0 0 0.004" quat="0.707107 -0.707107 0 0" mesh="base_coupling"/>',
        '                      <geom class="visual_gripper" pos="0 0 0.0108" quat="1 0 0 0" material="metal" mesh="c-a01-85-open"/>',
        '                      <site name="gripper" pos="0 0 0.1489"/>',
        '                      <geom name="hand_capsule" class="collision_gripper" type="capsule" conaffinity="1" size="0.04 0.06" rgba="1 1 1 0.3" pos="0 0 0.01"/>',
        '                      <!-- Left-hand side 4-bar linkage -->',
        '                      <body name="left_driver" pos="-0.0306011 0.00475 0.0657045" quat="0.707107 -0.707107 0 0" gravcomp="1">',
        '                        <inertial mass="0.00899563" pos="0 0.0177547 0.00107314" quat="0.681301 0.732003 0 0" diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>',
        '                        <joint name="left_driver_joint" class="driver"/>',
        '                        <geom class="visual_gripper" pos="0.0306011 0.0549045 -0.0047" quat="0.707107 0.707107 0 0" material="metal" mesh="driver"/>',
        '                        <body name="left_coupler" pos="-0.0314249 0.00453223 -0.0102" quat="8.02038e-06 0 0 1" gravcomp="1">',
        '                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636" diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>',
        '                          <geom class="visual_gripper" pos="-0.062026 -0.0503723 0.0055" quat="0.707107 -0.707107 0 0" mesh="coupler"/>',
        '                          <geom name="left_coupler_col_1" class="pad_box2" pos="0.005 0.025 0.01" quat="1 1 -0.1 0" type="capsule" size="0.009 0.02"/>',
        '                        </body>',
        '                      </body>',
        '                      <body name="left_spring_link" pos="-0.0127 -0.012 0.07222" quat="0.707107 -0.707107 -4.97726e-06 -4.97726e-06" gravcomp="1">',
        '                        <inertial mass="0.0221642" pos="-8.65005e-09 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403" diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>',
        '                        <joint name="left_spring_link_joint" class="spring_link"/>',
        '                        <geom class="visual_gripper" pos="0.0127 0.06142 0.01205" quat="0.707107 0.707107 0 0" type="mesh" mesh="spring_link"/>',
        '                        <body name="left_follower" pos="-0.0382079 -0.0425003 0.00295" quat="0 -1 -1.90231e-05 0" gravcomp="1">',
        '                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0" diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>',
        '                          <joint name="left_follower" class="follower"/>',
        '                          <geom class="visual_gripper" pos="0.0509079 -0.10392 -0.0091" quat="0.707107 -0.707107 0 0" type="mesh" mesh="follower"/>',
        '                          <geom class="visual_gripper" pos="0.0509079 -0.10392 -0.0091" quat="0.707107 -0.707107 0 0" type="mesh" material="metal" mesh="tongue"/>',
        '                          <geom name="left_follower_pad2" class="pad_box2" type="capsule" size="0.009 0.012 0.008" pos="-0.0035 -0.002 -0.009" quat="1 1 0 0"/>',
        '                          <body name="left_pad" pos="-0.0377897 -0.103916 -0.0091" quat="0.707107 -0.707107 3.16527e-05 -3.16527e-05" gravcomp="1">',
        '                            <geom class="pad_box2" name="left_finger_pad"/>',
        '                          </body>',
        '                        </body>',
        '                      </body>',
        '                      <!-- Right-hand side 4-bar linkage -->',
        '                      <body name="right_driver" pos="0.0306011 -0.00475 0.0657045" quat="0 0 -0.707107 0.707107" gravcomp="1">',
        '                        <inertial mass="0.00899563" pos="2.96931e-12 0.0177547 0.00107314" quat="0.681301 0.732003 0 0" diaginertia="1.72352e-06 1.60906e-06 3.22006e-07"/>',
        '                        <joint name="right_driver_joint" class="driver"/>',
        '                        <geom class="visual_gripper" pos="0.0306011 0.0549045 -0.0047" quat="0.707107 0.707107 0 0" material="metal" mesh="driver"/>',
        '                        <body name="right_coupler" pos="-0.0314249 0.00453223 -0.0102" quat="0 0 0 1" gravcomp="1">',
        '                          <inertial mass="0.0140974" pos="0 0.00301209 0.0232175" quat="0.705636 -0.0455904 0.0455904 0.705636" diaginertia="4.16206e-06 3.52216e-06 8.88131e-07"/>',
        '                          <geom class="visual_gripper" pos="-0.062026 -0.0503723 0.0055" quat="0.707107 -0.707107 0 0" mesh="coupler"/>',
        '                          <geom name="right_coupler_col_1" class="pad_box2" pos="0.005 0.025 0.01" quat="1 1 -0.1 0" type="capsule" size="0.009 0.02"/>',
        '                        </body>',
        '                      </body>',
        '                      <body name="right_spring_link" pos="0.0127 0.012 0.07222" quat="0 0 -0.707107 0.707107" gravcomp="1">',
        '                        <inertial mass="0.0221642" pos="-8.65005e-09 0.0181624 0.0212658" quat="0.663403 -0.244737 0.244737 0.663403" diaginertia="8.96853e-06 6.71733e-06 2.63931e-06"/>',
        '                        <joint name="right_spring_link_joint" class="spring_link"/>',
        '                        <geom class="visual_gripper" pos="0.0127 0.06142 0.01205" quat="0.707107 0.707107 0 0" mesh="spring_link"/>',
        '                        <body name="right_follower" pos="-0.0382079 -0.0425003 0.00295" quat="0 -1 1.79721e-11 0" gravcomp="1">',
        '                          <inertial mass="0.0125222" pos="0 -0.011046 0.0124786" quat="1 0.1664 0 0" diaginertia="2.67415e-06 2.4559e-06 6.02031e-07"/>',
        '                          <joint name="right_follower_joint" class="follower"/>',
        '                          <geom class="visual_gripper" pos="0.0509079 -0.10392 -0.0091" quat="0.707107 -0.707107 0 0" material="metal" mesh="tongue"/>',
        '                          <geom class="visual_gripper" pos="0.0509079 -0.10392 -0.0091" quat="0.707107 -0.707107 0 0" mesh="follower"/>',
        '                          <geom name="right_follower_pad2" class="pad_box2" type="capsule" size="0.009 0.012 0.008" pos="-0.0035 -0.002 -0.009" quat="1 1 0 0"/>',
        '                          <body name="right_pad" pos="-0.0377897 -0.103916 -0.0091" quat="0.707107 -0.707107 3.16527e-05 -3.16527e-05" gravcomp="1">',
        '                            <geom class="pad_box2" name="right_finger_pad"/>',
        '                          </body>',
        '                        </body>',
        '                      </body>',
        '                    </body>',
        '                  </body>',
        '                </body>',
        '              </body>',
        '            </body>',
        '          </body>',
        '        </body>',
        '      </body>',
        '    </body>',
        '',
    ]


def _generate_contacts() -> List[str]:
    """Generate contact exclusion section."""
    return [
        '  <!-- ============== CONTACT EXCLUSIONS ============== -->',
        '  <contact>',
        '    <exclude body1="robotiq_base" body2="left_driver"/>',
        '    <exclude body1="robotiq_base" body2="right_driver"/>',
        '    <exclude body1="robotiq_base" body2="left_spring_link"/>',
        '    <exclude body1="robotiq_base" body2="right_spring_link"/>',
        '    <exclude body1="right_coupler" body2="right_follower"/>',
        '    <exclude body1="left_coupler" body2="left_follower"/>',
        '  </contact>',
        '',
    ]


def _generate_equality() -> List[str]:
    """Generate equality constraint section."""
    return [
        '  <!-- ============== CONSTRAINTS ============== -->',
        '  <equality>',
        '    <connect anchor="-0.0179014 -0.00651468 0.0044" body1="right_follower" body2="right_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
        '    <connect anchor="-0.0179014 -0.00651468 0.0044" body1="left_follower" body2="left_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
        '    <joint joint1="right_driver_joint" joint2="left_driver_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1"/>',
        '  </equality>',
        '',
    ]


def _generate_actuators() -> List[str]:
    """Generate actuator section."""
    return [
        '  <!-- ============== ACTUATORS ============== -->',
        '  <actuator>',
        '    <position class="panda" name="actuator1" joint="joint1" kp="1000" kv="20" ctrlrange="-2.8973 2.8973"/>',
        '    <position class="panda" name="actuator2" joint="joint2" kp="1000" kv="20" ctrlrange="-1.7628 1.7628"/>',
        '    <position class="panda" name="actuator3" joint="joint3" kp="750" kv="4" ctrlrange="-2.8973 2.8973"/>',
        '    <position class="panda" name="actuator4" joint="joint4" kp="750" kv="4" ctrlrange="-3.0718 -0.0698"/>',
        '    <position class="panda" name="actuator5" joint="joint5" kp="300" kv="2" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>',
        '    <position class="panda" name="actuator6" joint="joint6" kp="300" kv="2" forcerange="-12 12" ctrlrange="-0.0175 3.7525"/>',
        '    <position class="panda" name="actuator7" joint="joint7" kp="300" kv="2" forcerange="-12 12" ctrlrange="-2.8973 2.8973"/>',
        '    <position class="2f85" name="fingers_actuator" joint="left_driver_joint" forcerange="-5 5" ctrlrange="0 0.82" kp="10" kv="1"/>',
        '  </actuator>',
        '',
    ]


def _generate_sensors() -> List[str]:
    """Generate sensor section."""
    return [
        '  <!-- ============== SENSORS ============== -->',
        '  <sensor>',
        '    <jointpos name="joint1_pos" joint="joint1"/>',
        '    <jointpos name="joint2_pos" joint="joint2"/>',
        '    <jointpos name="joint3_pos" joint="joint3"/>',
        '    <jointpos name="joint4_pos" joint="joint4"/>',
        '    <jointpos name="joint5_pos" joint="joint5"/>',
        '    <jointpos name="joint6_pos" joint="joint6"/>',
        '    <jointpos name="joint7_pos" joint="joint7"/>',
        '    <jointpos name="gripper_pos" joint="left_driver_joint"/>',
        '    <framepos name="gripper_pos_world" objtype="site" objname="gripper"/>',
        '  </sensor>',
        '',
    ]


def _generate_keyframe(
    objects: List[ObjectInfo],
    include_robot: bool,
    object_positions: List[Tuple[float, float, float]],
) -> List[str]:
    """Generate keyframe section."""
    lines = ['  <!-- ============== KEYFRAME ============== -->', '  <keyframe>']

    # Robot joint positions (home pose)
    robot_qpos = "0 -0.785 0 -2.356 0 1.571 0.785 0 0 0 0 0 0" if include_robot else ""
    robot_ctrl = "0 -0.785 0 -2.356 0 1.571 0.785 0" if include_robot else ""

    # Object positions (free joints: x y z qw qx qy qz)
    obj_qpos_list = []
    for i, obj in enumerate(objects):
        x_pos, y_pos, z_pos = object_positions[i]
        obj_qpos_list.append(f"{x_pos:.3f} {y_pos:.3f} {z_pos:.3f} 1 0 0 0")

    all_qpos = robot_qpos
    if obj_qpos_list:
        all_qpos += "\n               " + "\n               ".join(obj_qpos_list)

    lines.extend([
        f'    <key name="home"',
        f'         qpos="{all_qpos}"',
    ])

    if include_robot:
        lines.append(f'         ctrl="{robot_ctrl}"/>')
    else:
        lines.append('         />')

    lines.extend(['  </keyframe>', ''])
    return lines


def convert_dataset(
    dataset_dir: str,
    output_dir: str,
    scene_name: Optional[str] = None,
    include_robot: bool = True,
    table_size: Tuple[float, float] = (1.0, 0.7),
    table_height: float = 0.75,
    copy_robot_meshes: bool = True,
) -> Dict:
    """
    Convert a dataset to MuJoCo format.

    Args:
        dataset_dir: Path to dataset directory
        output_dir: Output directory for converted files
        scene_name: Name for the scene (default: dataset directory name)
        include_robot: Whether to include Franka robot
        table_size: (width, depth) of table
        table_height: Height of table surface
        copy_robot_meshes: Whether to copy robot mesh files

    Returns:
        Dictionary with conversion results
    """
    # Validate input
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Determine scene name
    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(dataset_dir)).replace('-', '_')

    # Create output directories
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Scan dataset
    print(f"Scanning dataset: {dataset_dir}")
    objects = scan_dataset(dataset_dir)
    print(f"Found {len(objects)} objects:")
    for obj in objects:
        print(f"  - {obj.name}: {obj.size[0]*100:.1f}cm x {obj.size[1]*100:.1f}cm x {obj.size[2]*100:.1f}cm ({obj.vertex_count} vertices)")

    # Copy object meshes
    print(f"\nCopying meshes to: {meshes_dir}")
    for obj in objects:
        dest = os.path.join(meshes_dir, f"{obj.name}.obj")
        shutil.copy2(obj.obj_path, dest)
        print(f"  Copied: {obj.name}.obj")
        if convex_decompose_mesh is None:
            raise RuntimeError(
                "Convex decomposition unavailable: failed to import mesh2mjcf.convex_decompose_mesh."
            )
        parts = convex_decompose_mesh(Path(dest), Path(meshes_dir), obj.name, scene_mode=False)
        if parts <= 0:
            raise RuntimeError(f"Convex decomposition produced no parts for {obj.name}.")
        obj.convex_parts = parts
        print(f"  Convex decomposition: {parts} parts")

    # Copy robot meshes if needed
    if include_robot and copy_robot_meshes:
        # Check if robot meshes already exist
        franka_dir = os.path.join(meshes_dir, "franka")
        robotiq_dir = os.path.join(meshes_dir, "robotiq")

        if not os.path.exists(franka_dir) or not os.path.exists(robotiq_dir):
            # Try to find robot meshes from existing vlm_rl assets
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vlm_rl_dir = os.path.dirname(script_dir)
            existing_franka = os.path.join(vlm_rl_dir, "assets", "meshes", "franka")
            existing_robotiq = os.path.join(vlm_rl_dir, "assets", "meshes", "robotiq")

            if os.path.exists(existing_franka) and os.path.exists(existing_robotiq):
                if not os.path.exists(franka_dir):
                    shutil.copytree(existing_franka, franka_dir)
                    print(f"  Copied: franka/ directory")
                if not os.path.exists(robotiq_dir):
                    shutil.copytree(existing_robotiq, robotiq_dir)
                    print(f"  Copied: robotiq/ directory")
            else:
                print("  Warning: Robot meshes not found. Please copy franka/ and robotiq/ directories manually.")

    # Generate MJCF
    print(f"\nGenerating MJCF scene...")
    mjcf_path = generate_mjcf(
        objects=objects,
        output_dir=output_dir,
        scene_name=scene_name,
        include_robot=include_robot,
        table_size=table_size,
        table_height=table_height,
    )
    print(f"  Generated: {mjcf_path}")

    # Save metadata
    metadata = {
        "dataset_dir": dataset_dir,
        "scene_name": scene_name,
        "objects": [
            {
                "name": obj.name,
                "size": obj.size.tolist(),
                "center": obj.center.tolist(),
                "vertex_count": obj.vertex_count,
                "convex_parts": obj.convex_parts,
            }
            for obj in objects
        ],
        "include_robot": include_robot,
        "table_size": list(table_size),
        "table_height": table_height,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    return {
        "mjcf_path": mjcf_path,
        "meshes_dir": meshes_dir,
        "metadata_path": metadata_path,
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset to MuJoCo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_dataset.py /path/to/dataset --output ./converted_scene
  python convert_dataset.py /path/to/dataset --output ./scene --name my_scene --no-robot
        """
    )
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: ./assets/<dataset_name>)")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Scene name (default: dataset directory name)")
    parser.add_argument("--no-robot", action="store_true",
                        help="Don't include Franka robot in scene")
    parser.add_argument("--table-width", type=float, default=1.0,
                        help="Table width in meters (default: 1.0)")
    parser.add_argument("--table-depth", type=float, default=0.7,
                        help="Table depth in meters (default: 0.7)")
    parser.add_argument("--table-height", type=float, default=0.75,
                        help="Table height in meters (default: 0.75)")
    parser.add_argument("--test", action="store_true",
                        help="Test loading the generated scene")

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vlm_rl_dir = os.path.dirname(script_dir)
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir)).replace('-', '_')
        args.output = os.path.join(vlm_rl_dir, "assets", dataset_name)

    # Convert dataset
    result = convert_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output,
        scene_name=args.name,
        include_robot=not args.no_robot,
        table_size=(args.table_width, args.table_depth),
        table_height=args.table_height,
    )

    print(f"\n{'='*50}")
    print("Conversion complete!")
    print(f"  Output directory: {args.output}")
    print(f"  MJCF file: {result['mjcf_path']}")

    # Test loading if requested
    if args.test:
        print(f"\nTesting scene loading...")
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_path(result['mjcf_path'])
            data = mujoco.MjData(model)
            print(f"  Success! Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
