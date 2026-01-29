#!/usr/bin/env python3
"""
Mesh to MJCF Converter with Convex Decomposition

Convert OBJ/STL mesh files to MuJoCo MJCF format with optional convex decomposition
for accurate collision detection.

Features:
- Convex decomposition using CoACD for precise collision geometry
- MTL material and texture support
- Multi-material OBJ file handling
- Automatic scene generation with Franka robot

Usage:
    # Basic conversion
    python scripts/mesh2mjcf.py /path/to/meshes_dir

    # With convex decomposition
    python scripts/mesh2mjcf.py /path/to/meshes_dir -cd

    # High-precision scene-level decomposition
    python scripts/mesh2mjcf.py /path/to/meshes_dir -cd --scene
"""

import os
import shutil
import argparse
import xml.etree.ElementTree as ET
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Default output directory (relative to this script)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR.parent / "assets"
_DEFAULT_TABLE_POS = [-0.5, 0.0, 0.55]
_DEFAULT_TABLE_SIZE = [0.4, 0.6, 0.01]


def convex_decompose_mesh(input_file_path: Path, output_dir: Path,
                          asset_name: str, scene_mode: bool = False) -> int:
    """Run convex decomposition and export part meshes.

    Args:
        input_file_path: Source mesh file (.obj/.stl)
        output_dir: Output directory to write *_part_*.obj files
        asset_name: Base name for part files
        scene_mode: Use higher precision if True

    Returns:
        Number of convex parts generated
    """
    import coacd
    import trimesh

    mesh = trimesh.load(str(input_file_path), force="mesh")
    mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
    threshold = 0.01 if scene_mode else 0.05
    parts = coacd.run_coacd(mesh_coacd, threshold=threshold)

    for i, part in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
        part_path = output_dir / f"{asset_name}_part_{i}.obj"
        part_mesh.export(str(part_path))

    return len(parts)


# --- MTL Material Processing ---

MTL_FIELDS = (
    "Ka",       # Ambient color
    "Kd",       # Diffuse color
    "Ks",       # Specular color
    "d",        # Transparency (alpha)
    "Tr",       # 1 - transparency
    "Ns",       # Shininess
    "map_Kd",   # Diffuse texture map
)


@dataclass
class Material:
    """Material properties from MTL file."""

    name: str
    Ka: Optional[str] = None
    Kd: Optional[str] = None
    Ks: Optional[str] = None
    d: Optional[str] = None
    Tr: Optional[str] = None
    Ns: Optional[str] = None
    map_Kd: Optional[str] = None

    @staticmethod
    def from_string(lines: Sequence[str]) -> "Material":
        """Construct a Material object from MTL file lines."""
        attrs = {"name": lines[0].split(" ")[1].strip()}
        for line in lines[1:]:
            for attr in MTL_FIELDS:
                if line.startswith(attr):
                    elems = line.split(" ")[1:]
                    elems = [elem for elem in elems if elem != ""]
                    attrs[attr] = " ".join(elems)
                    break
        return Material(**attrs)

    def mjcf_rgba(self) -> str:
        """Convert material properties to MJCF RGBA string."""
        Kd = self.Kd or "1.0 1.0 1.0"
        if self.d is not None:
            alpha = self.d
        elif self.Tr is not None:
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        """Convert shininess value to MJCF format."""
        if self.Ns is not None:
            Ns = float(self.Ns) / 1_000
        else:
            Ns = 0.5
        return f"{Ns}"

    def mjcf_specular(self) -> str:
        """Convert specular value to MJCF format."""
        if self.Ks is not None:
            Ks = sum(list(map(float, self.Ks.split(" ")))) / 3
        else:
            Ks = 0.5
        return f"{Ks}"


def parse_mtl_name(lines: Sequence[str]) -> Optional[str]:
    """Parse MTL file name from OBJ file lines."""
    mtl_regex = re.compile(r"^mtllib\s+(.+?\.mtl)(?:\s*#.*)?\s*\n?$")
    for line in lines:
        match = mtl_regex.match(line)
        if match is not None:
            return match.group(1)
    return None


def copy_obj_with_mtl(obj_source: Path, obj_target: Path) -> None:
    """Copy OBJ file and its associated MTL file."""
    obj_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(obj_source, obj_target)

    try:
        with open(obj_source, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.strip().startswith('mtllib '):
                mtl_filename = line.strip().split()[1]
                mtl_source = obj_source.parent / mtl_filename
                mtl_target = obj_target.parent / mtl_filename

                if mtl_source.exists():
                    shutil.copy2(mtl_source, mtl_target)
                    logging.info(f"Copied MTL file: {mtl_source.name}")
                break
    except Exception as e:
        logging.warning(f"Failed to copy MTL file: {e}")


def parse_mtl_file(mtl_path: Path) -> Dict[str, Material]:
    """Parse MTL file and return material dictionary."""
    materials = {}

    if not mtl_path.exists():
        return materials

    with open(mtl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith("#")]
    lines = [line for line in lines if line.strip()]
    lines = [line.strip() for line in lines]

    sub_mtls = []
    for line in lines:
        if line.startswith("newmtl"):
            sub_mtls.append([])
        if sub_mtls:
            sub_mtls[-1].append(line)

    for sub_mtl in sub_mtls:
        if sub_mtl:
            material = Material.from_string(sub_mtl)
            materials[material.name] = material

    return materials


def split_obj_by_materials(obj_path: Path, output_dir: Path) -> Tuple[Dict[str, Material], List[str]]:
    """Split OBJ file by materials.

    Returns:
        (materials_dict, submesh_files): Material dictionary and submesh file list
    """
    materials = {}
    submesh_files = []

    with open(obj_path, 'r', encoding='utf-8') as f:
        obj_lines = f.readlines()

    mtl_name = parse_mtl_name(obj_lines)
    if mtl_name:
        mtl_path = obj_path.parent / mtl_name
        materials = parse_mtl_file(mtl_path)

    if len(materials) <= 1:
        return materials, []

    try:
        import trimesh

        mesh = trimesh.load(
            obj_path,
            split_object=True,
            group_material=True,
            process=False,
            maintain_order=False,
        )

        if isinstance(mesh, trimesh.base.Trimesh):
            target_file = output_dir / f"{obj_path.stem}.obj"
            shutil.copy(obj_path, target_file)
            return materials, []
        else:
            obj_stem = obj_path.stem
            logging.info(f"Splitting OBJ by material: {len(mesh.geometry)} submeshes")

            for i, (material_name, geom) in enumerate(mesh.geometry.items()):
                submesh_file = f"{obj_stem}_{i}.obj"
                submesh_path = output_dir / submesh_file

                geom.visual.material.name = material_name
                geom.export(str(submesh_path), include_texture=True, header=None)
                submesh_files.append(submesh_file)

                logging.info(f"  Saved submesh: {submesh_file} (material: {material_name})")

            temp_mtl = output_dir / "material.mtl"
            if temp_mtl.exists():
                temp_mtl.unlink()

            return materials, submesh_files

    except ImportError:
        logging.warning("trimesh not installed, skipping material split")
        return materials, []
    except Exception as e:
        logging.warning(f"Failed to split OBJ by materials: {e}")
        return materials, []


# --- XML Generation ---

def create_asset_xml(meshes_dir: str, asset_name: str, convex_parts: Optional[int] = None,
                     materials: Optional[Dict[str, Material]] = None,
                     submesh_files: Optional[List[str]] = None) -> ET.Element:
    """Create asset dependency XML."""
    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")

    # Add MTL materials
    if materials:
        for material_name, material in materials.items():
            material_elem = ET.SubElement(asset, "material")
            material_elem.set("name", f"{asset_name}_{material_name}")
            material_elem.set("rgba", material.mjcf_rgba())
            material_elem.set("specular", material.mjcf_specular())
            material_elem.set("shininess", material.mjcf_shininess())

            if material.map_Kd:
                texture_elem = ET.SubElement(asset, "texture")
                texture_elem.set("type", "2d")
                texture_elem.set("name", f"{asset_name}_{material_name}_texture")
                texture_elem.set("file", f"{meshes_dir}/{asset_name}/{material.map_Kd}")

                material_elem.set("texture", f"{asset_name}_{material_name}_texture")
                material_elem.attrib.pop("rgba", None)

    # Add submesh files (for multi-material OBJ)
    if submesh_files:
        for submesh_file in submesh_files:
            submesh_name = submesh_file.replace('.obj', '')
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", submesh_name)
            part_mesh.set("file", f"{meshes_dir}/{asset_name}/{submesh_file}")

    # Add convex decomposition parts
    if convex_parts:
        for i in range(convex_parts):
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", f"{asset_name}_part_{i}")
            part_mesh.set("file", f"{meshes_dir}/{asset_name}_part_{i}.obj")

    return root


def create_geom_xml(asset_name: str, mass: float, diaginertia: List[float], rgba: List[float],
                    free_joint: bool = False, convex_parts: Optional[int] = None,
                    materials: Optional[Dict[str, Material]] = None,
                    submesh_files: Optional[List[str]] = None,
                    output_dir: Path = _DEFAULT_OUTPUT_DIR) -> ET.Element:
    """Create geometry definition XML."""
    root = ET.Element("mujocoinclude")

    if free_joint:
        joint_elem = ET.SubElement(root, "joint")
        joint_elem.set("type", "free")

    # Handle multi-material case
    if submesh_files and materials:
        for i, submesh_file in enumerate(submesh_files):
            submesh_name = submesh_file.replace('.obj', '')
            geom_elem = ET.SubElement(root, "geom")
            geom_elem.set("type", "mesh")
            geom_elem.set("mesh", submesh_name)

            material_assigned = False
            submesh_path = output_dir / "meshes" / asset_name / submesh_file
            if submesh_path.exists():
                try:
                    with open(submesh_path, 'r', encoding='utf-8') as f:
                        submesh_lines = f.readlines()

                    for line in submesh_lines:
                        line = line.strip()
                        if line.startswith('usemtl '):
                            mtl_name = line.split()[1]
                            material_name = f"{asset_name}_{mtl_name}"
                            geom_elem.set("material", material_name)
                            material_assigned = True
                            break
                except Exception as e:
                    logging.warning(f"Cannot read submesh file {submesh_path}: {e}")

            if not material_assigned:
                geom_elem.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

    elif materials and len(materials) == 1:
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("mesh", asset_name)

        material_name = list(materials.keys())[0]
        geom_elem.set("material", f"{asset_name}_{material_name}")

    elif convex_parts:
        # Visual mesh (non-colliding)
        visual_geom = ET.SubElement(root, "geom")
        visual_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        visual_geom.set("mesh", asset_name)
        visual_geom.set("contype", "0")
        visual_geom.set("conaffinity", "0")

        # Collision meshes (convex parts)
        for i in range(convex_parts):
            collision_geom = ET.SubElement(root, "geom")
            collision_geom.set("type", "mesh")
            collision_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} 0")  # Invisible
            collision_geom.set("mesh", f"{asset_name}_part_{i}")
    else:
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        geom_elem.set("mesh", asset_name)

    # Add convex collision parts for multi-material case too
    if convex_parts and (submesh_files or (materials and len(materials) == 1)):
        for i in range(convex_parts):
            collision_geom = ET.SubElement(root, "geom")
            collision_geom.set("type", "mesh")
            collision_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} 0")
            collision_geom.set("mesh", f"{asset_name}_part_{i}")

    return root



def save_xml_with_formatting(root: ET.Element, filepath: str) -> None:
    """Save formatted XML file."""
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8', xml_declaration=False)


def create_preview_xml(asset_name: str, meshes_dir: str) -> ET.Element:
    """Create preview XML file."""
    root = ET.Element("mujoco")
    root.set("model", "preview")

    option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")

    compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", f"../{meshes_dir}")
    compiler.set("texturedir", f"../{meshes_dir}/")

    include = ET.SubElement(root, "include")
    include.set("file", f"{asset_name}_dependencies.xml")

    default = ET.SubElement(root, "default")
    obj_default = ET.SubElement(default, "default")
    geom_default = ET.SubElement(obj_default, "geom")
    geom_default.set("group", "2")
    geom_default.set("type", "mesh")
    geom_default.set("contype", "0")
    geom_default.set("conaffinity", "0")

    worldbody = ET.SubElement(root, "worldbody")

    floor_geom = ET.SubElement(worldbody, "geom")
    floor_geom.set("name", "floor")
    floor_geom.set("type", "plane")
    floor_geom.set("size", "2 2 0.1")
    floor_geom.set("rgba", ".8 .8 .8 1")

    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 3")
    light.set("dir", "0 0 -1")

    body = ET.SubElement(worldbody, "body")
    body.set("name", asset_name)
    body.set("pos", "0 0 0.5")

    body_include = ET.SubElement(body, "include")
    body_include.set("file", f"{asset_name}.xml")

    return root


def _format_vec(values: Sequence[float]) -> str:
    return " ".join(f"{v:g}" for v in values)


def generate_scene_template(body_definitions: List[str], dependency_includes: List[str],
                            obj_count: int, scene_name: str,
                            table_pos: Sequence[float], table_size: Sequence[float]) -> str:
    """Generate complete MuJoCo scene template with Franka robot."""

    robot_qpos = "0 -0.76219644 0 -2.75629741 0 1.96083466 0 0 0 0 0 0 0"

    obj_qpos_list = []
    for i in range(obj_count):
        pos_y = i * 0.2
        obj_qpos_list.append(f"-0.4 {pos_y:.2f} 0.85 1 0 0 0")

    objs_qpos_str = "\n".join(obj_qpos_list)
    full_qpos = f"{robot_qpos}\n{objs_qpos_str}\n"

    dependency_includes_text = "\n            ".join(dependency_includes)
    bodies_text = "\n            ".join(body_definitions)

    template = f"""<mujoco model="franka_pick_{scene_name}">
    <include file="panda_robotiq.xml"/>
    <statistic center="0.3 0 0.4" extent="1"/>

    <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
        <flag eulerdamp="disable"/>
    </option>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
        <scale contactwidth="0.075" contactheight="0.025" forcewidth="0.05" com="0.05" framewidth="0.01" framelength="0.2"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    {dependency_includes_text}

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <camera fovy="42" name="front" pos="-1.302 0.035 1.354" xyaxes="-0.029 -1.000 -0.000 0.592 -0.017 0.806"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
        <geom name="table" pos="{_format_vec(table_pos)}" size="{_format_vec(table_size)}" type="box" contype="1" rgba="0.6 0.6 0.6 1"/>

        {bodies_text}

        <body mocap="true" name="mocap_target">
            <site name="mocap_target_site" size='0.001' type='sphere'/>
            <geom name="mocap_target_box" type="box" size="0.02 0.02 0.02" density="0" contype="0" conaffinity="0" rgba="0.3 0.6 0.3 0.2"/>
            <geom type="cylinder" pos="0.02 0 0" euler="0 1.5708 0" size=".003 0.02" density="0" contype="0" conaffinity="0" rgba="1 0 0 .2"/>
            <geom type="cylinder" pos="0 0.02 0" euler="1.5708 0 0" size=".003 0.02" density="0" contype="0" conaffinity="0" rgba="0 1 0 .2"/>
            <geom type="cylinder" pos="0 0 0.02" euler="0 0 0"      size=".003 0.02" density="0" contype="0" conaffinity="0" rgba="0 0 1 .2"/>
        </body>
    </worldbody>

    <keyframe>
        <key name="home" qpos="{full_qpos}" ctrl="0 -0.76219644 0 -2.75629741 0 1.96083466 0 0"/>
    </keyframe>
</mujoco>
"""
    return template


def main():
    parser = argparse.ArgumentParser(
        description="Convert mesh files to MuJoCo MJCF format with convex decomposition."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing mesh files (.obj or .stl)")
    parser.add_argument("--rgba", nargs=4, type=float, default=[0.5, 0.5, 0.5, 1],
                        help="RGBA color for meshes (default: 0.5 0.5 0.5 1)")
    parser.add_argument("--mass", type=float, default=0.001,
                        help="Mass in kg (default: 0.001)")
    parser.add_argument("-o", "--output", type=str, default=str(_DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--diaginertia", nargs=3, type=float, default=[0.00002, 0.00002, 0.00002],
                        help="Diagonal inertia tensor (default: 0.00002 0.00002 0.00002)")
    parser.add_argument("--free_joint", action="store_true",
                        help="Add free joint for movable objects")
    parser.add_argument("-cd", "--convex_decomposition", action="store_true",
                        help="Enable convex decomposition using CoACD")
    parser.add_argument("--scene", action="store_true",
                        help="Use high-precision scene-level decomposition (threshold=0.01)")
    parser.add_argument("--verbose", action="store_true",
                        help="Launch MuJoCo viewer after conversion")
    parser.add_argument("--test", action="store_true",
                        help="Test loading the generated scene")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    scene_name = input_dir.name

    # Find all mesh files
    mesh_files = list(input_dir.glob("*.obj")) + list(input_dir.glob("*.stl"))

    # Filter out background files and files with "final" in name
    mesh_files = [f for f in mesh_files if not f.stem.startswith("bg_")]
    mesh_files = [f for f in mesh_files if "final" not in f.stem.lower()]

    if not mesh_files:
        logging.error(f"No .obj or .stl files found in {input_dir}")
        return 1

    logging.info(f"Found {len(mesh_files)} mesh files in {input_dir}")

    # Prepare output directories
    meshes_output_dir = output_dir / "meshes"
    mjcf_output_dir = output_dir / "mjcf"
    meshes_output_dir.mkdir(parents=True, exist_ok=True)
    mjcf_output_dir.mkdir(parents=True, exist_ok=True)

    body_definitions = []
    # Process each mesh file
    for input_file_path in mesh_files:
        asset_name = input_file_path.stem
        logging.info(f"\n>>> Processing: {asset_name}")

        # Create output directory for this asset
        asset_mesh_dir = meshes_output_dir / asset_name
        asset_mesh_dir.mkdir(parents=True, exist_ok=True)

        # Copy mesh files
        if input_file_path.suffix == ".obj":
            copy_obj_with_mtl(input_file_path, asset_mesh_dir / input_file_path.name)
        else:
            shutil.copy(input_file_path, asset_mesh_dir)

        # Handle materials
        materials = {}
        submesh_files = []
        if input_file_path.suffix == ".obj":
            obj_path = asset_mesh_dir / f"{asset_name}.obj"
            materials, submesh_files = split_obj_by_materials(obj_path, asset_mesh_dir)

            # Copy texture files
            for mat in materials.values():
                if mat.map_Kd:
                    tex_src = input_file_path.parent / mat.map_Kd
                    if tex_src.exists():
                        shutil.copy(tex_src, asset_mesh_dir / mat.map_Kd)

        # Convex decomposition
        convex_parts_count = 0
        if args.convex_decomposition:
            try:
                convex_parts_count = convex_decompose_mesh(
                    input_file_path, asset_mesh_dir, asset_name, scene_mode=args.scene
                )
                threshold = 0.01 if args.scene else 0.05
                logging.info(f"  Convex decomposition: {convex_parts_count} parts (threshold={threshold})")
            except ImportError as e:
                logging.error(f"Convex decomposition requires coacd and trimesh: {e}")
                logging.error("Install with: pip install coacd trimesh")
                return 1

        # Generate XML files
        asset_xml = create_asset_xml(
            "meshes", asset_name,
            convex_parts_count if args.convex_decomposition else None,
            materials if (submesh_files or len(materials) == 1) else None,
            submesh_files if submesh_files else None
        )
        save_xml_with_formatting(asset_xml, str(mjcf_output_dir / f"{asset_name}_dependencies.xml"))

        geom_xml = create_geom_xml(
            asset_name, args.mass, args.diaginertia, args.rgba, args.free_joint,
            convex_parts_count if args.convex_decomposition else None,
            materials if (submesh_files or len(materials) == 1) else None,
            submesh_files if submesh_files else None,
            output_dir
        )
        save_xml_with_formatting(geom_xml, str(mjcf_output_dir / f"{asset_name}.xml"))

    # Generate scene XML
    dependency_includes = []
    final_obj_count = 0

    for filename in sorted(os.listdir(mjcf_output_dir)):
        if filename.endswith('.xml'):
            if '_dependencies' in filename:
                dependency_includes.append(f'<include file="mjcf/{filename}"/>')
            elif not filename.startswith(('pick_', 'preview', 'bg_')):
                obj_name = filename.replace('.xml', '')
                pos_y = final_obj_count * 0.2
                body_entry = f"""
        <body name="{obj_name}" pos="-0.4 {pos_y:.2f} 0.85" quat="1 0 0 0">
            <joint type="free" frictionloss="0.01"/>
            <site group="4" name="{obj_name}_site" type="box" pos="0 0 0" size="0.01 0.025 0.08" rgba="1 0 0 1"/>
            <include file="mjcf/{filename}"/>
        </body>"""
                body_definitions.append(body_entry)
                final_obj_count += 1

    # Save scene XML
    scene_xml = generate_scene_template(
        body_definitions, dependency_includes, final_obj_count, scene_name,
        _DEFAULT_TABLE_POS, _DEFAULT_TABLE_SIZE
    )
    scene_path = mjcf_output_dir / f"pick_{scene_name}.xml"
    with open(scene_path, "w", encoding='utf-8') as f:
        f.write(scene_xml)

    logging.info(f"\n{'='*60}")
    logging.info(f"Conversion complete! Processed {final_obj_count} objects.")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Scene file: {scene_path}")
    logging.info(f"{'='*60}")

    # Test loading
    if args.test or args.verbose:
        try:
            import mujoco
            logging.info("\nTesting scene loading...")
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            logging.info(f"Scene loaded successfully!")
            logging.info(f"  Bodies: {model.nbody}")
            logging.info(f"  Geoms: {model.ngeom}")
            logging.info(f"  Meshes: {model.nmesh}")

            if args.verbose:
                logging.info("\nLaunching MuJoCo viewer...")
                os.system(f"python -m mujoco.viewer --mjcf={scene_path}")
        except ImportError:
            logging.warning("MuJoCo not installed, skipping test")
        except Exception as e:
            logging.error(f"Failed to load scene: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
