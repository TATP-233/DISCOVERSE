import os
import shutil
import argparse
import xml.etree.ElementTree as ET
import logging
import multiprocessing as mp
from discoverse import DISCOVERSE_ASSETS_DIR

# 新增导入
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 使用示例 ---
"""
命令行使用示例:

单文件处理:
1. 基本转换 (使用默认颜色、质量、惯性，不固定，不分解):
    python scripts/mesh2mjcf.py /path/to/your/model.obj

2. 指定 RGBA 颜色:
    python scripts/mesh2mjcf.py /path/to/your/model.stl --rgba 0.8 0.2 0.2 1.0

3. 固定物体 (不能自由移动):
    python scripts/mesh2mjcf.py /path/to/your/model.obj --fix_joint

4. 进行凸分解 (用于更精确碰撞):
    python scripts/mesh2mjcf.py /path/to/your/model.obj -cd

5. 转换后立即用 MuJoCo 查看器预览:
    python scripts/mesh2mjcf.py /path/to/your/model.obj --verbose

6. 组合使用 (例如，进行凸分解并预览):
    python scripts/mesh2mjcf.py /path/to/your/model.obj -cd --verbose

7. 指定质量和惯性:
    python scripts/mesh2mjcf.py /path/to/your/model.obj --mass 0.5 --diaginertia 0.01 0.01 0.005

8. 多材质 OBJ 文件自动拆分:
    python scripts/mesh2mjcf.py /path/to/your/multi_material_model.obj
    # 脚本会自动检测 MTL 文件，按材质拆分 OBJ 文件，为每个材质创建独立的几何体

批量处理:
9. 批量转换目录中的所有文件 (自动使用多进程):
    python scripts/mesh2mjcf.py -d /path/to/mesh/directory

10. 批量转换并进行凸分解:
    python scripts/mesh2mjcf.py -d /path/to/mesh/directory -cd

11. 批量转换并指定进程数量:
    python scripts/mesh2mjcf.py -d /path/to/mesh/directory --processes 8

12. 批量转换并指定输出路径和材质:
    python scripts/mesh2mjcf.py -d /path/to/mesh/directory -o /output/path --rgba 0.8 0.2 0.2 1.0

注意:
- 脚本会自动检测带有 MTL 文件的 OBJ 模型，并按材质进行拆分
- 每个材质会在 MJCF 中生成对应的 material 定义
- 如果材质包含纹理贴图 (map_Kd)，会自动复制纹理文件并设置对应的 texture 和 material
- 材质拆分功能需要 trimesh 库支持
- 批量处理时，默认使用 CPU 核心数 - 2 个进程，可通过 --processes 参数调整
- 批量处理模式会自动启用多进程，根据 CPU 核心数自适应调整
"""

# --- MTL材质处理相关函数 ---

# MTL fields relevant to MuJoCo
MTL_FIELDS = (
    "Ka",   # Ambient color
    "Kd",   # Diffuse color
    "Ks",   # Specular color
    "d",    # Transparency (alpha)
    "Tr",   # 1 - transparency
    "Ns",   # Shininess
    "map_Kd",  # Diffuse texture map
)

@dataclass
class Material:
    """A convenience class for constructing MuJoCo materials from MTL files."""
    
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
        """Construct a Material object from a string."""
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
        if self.d is not None:  # alpha
            alpha = self.d
        elif self.Tr is not None:  # 1 - alpha
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        """Convert shininess value to MJCF format."""
        if self.Ns is not None:
            # Normalize Ns value to [0, 1]. Ns values normally range from 0 to 1000.
            Ns = float(self.Ns) / 1_000
        else:
            Ns = 0.5
        return f"{Ns}"

    def mjcf_specular(self) -> str:
        """Convert specular value to MJCF format."""
        if self.Ks is not None:
            # Take the average of the specular RGB values.
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
            name = match.group(1)
            return name
    return None

def copy_obj_with_mtl(obj_source: Path, obj_target: Path) -> None:
    """Copy OBJ file and its associated MTL file if it exists.
    
    Args:
        obj_source: Source OBJ file path
        obj_target: Target OBJ file path
    """
    # Copy the OBJ file
    obj_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(obj_source, obj_target)
    
    # Check for MTL file and copy it if it exists
    try:
        with open(obj_source, 'r') as f:
            lines = f.readlines()
        
        # Look for mtllib directive
        for line in lines:
            if line.strip().startswith('mtllib '):
                mtl_filename = line.strip().split()[1]
                mtl_source = obj_source.parent / mtl_filename
                mtl_target = obj_target.parent / mtl_filename
                
                if mtl_source.exists():
                    shutil.copy2(mtl_source, mtl_target)
                    print(f"Copied MTL file: {mtl_source} -> {mtl_target}")
                break
    except Exception as e:
        print(f"Warning: Failed to check/copy MTL file for {obj_source}: {e}")


def parse_mtl_file(mtl_path: Path) -> Dict[str, Material]:
    """解析 MTL 文件，返回材质字典"""
    materials = {}
    
    if not mtl_path.exists():
        return materials
    
    with open(mtl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove comments and empty lines
    lines = [line for line in lines if not line.startswith("#")]
    lines = [line for line in lines if line.strip()]
    lines = [line.strip() for line in lines]
    
    # Split at each new material definition
    sub_mtls = []
    for line in lines:
        if line.startswith("newmtl"):
            sub_mtls.append([])
        if sub_mtls:  # Only append if we have started a material
            sub_mtls[-1].append(line)
    
    # Process each material
    for sub_mtl in sub_mtls:
        if sub_mtl:  # Make sure the material has content
            material = Material.from_string(sub_mtl)
            materials[material.name] = material
    
    return materials


def split_obj_by_materials(obj_path: Path, output_dir: Path) -> Tuple[Dict[str, Material], List[str]]:
    """按材质拆分OBJ文件
    
    Args:
        obj_path: 输入的OBJ文件路径
        output_dir: 输出目录
        
    Returns:
        (materials_dict, submesh_files): 材质字典和子网格文件列表
    """
    materials = {}
    submesh_files = []
    
    # 读取OBJ文件内容
    with open(obj_path, 'r', encoding='utf-8') as f:
        obj_lines = f.readlines()
    
    # 解析MTL文件
    mtl_name = parse_mtl_name(obj_lines)
    if mtl_name:
        mtl_path = obj_path.parent / mtl_name
        materials = parse_mtl_file(mtl_path)
    
    if len(materials) <= 1:
        # 只有一个或没有材质，不需要拆分
        return materials, []
    
    try:
        # 使用 trimesh 按材质分组加载
        import trimesh
        
        mesh = trimesh.load(
            obj_path,
            split_object=True,
            group_material=True,
            process=False,
            maintain_order=False,
        )
        
        if isinstance(mesh, trimesh.base.Trimesh):
            # 单个网格，直接复制
            target_file = output_dir / f"{obj_path.stem}.obj"
            shutil.copy(obj_path, target_file)
            return materials, []
        else:
            # 多个子网格，分别保存
            obj_stem = obj_path.stem
            
            print(f"按材质拆分 OBJ 文件，共 {len(mesh.geometry)} 个子网格")
            
            for i, (material_name, geom) in enumerate(mesh.geometry.items()):
                submesh_file = f"{obj_stem}_{i}.obj"
                submesh_path = output_dir / submesh_file
                
                # 导出子网格
                geom.visual.material.name = material_name
                geom.export(str(submesh_path), include_texture=True, header=None)
                submesh_files.append(submesh_file)
                
                print(f"保存子网格: {submesh_file} (材质: {material_name})")
            
            # 删除生成的material.mtl文件（如果存在）
            temp_mtl = output_dir / "material.mtl"
            if temp_mtl.exists():
                temp_mtl.unlink()
                
            return materials, submesh_files
    
    except ImportError:
        print("警告: trimesh 未安装，无法按材质拆分OBJ文件")
        return materials, []
    except Exception as e:
        print(f"警告: 按材质拆分OBJ文件时出错: {e}")
        return materials, []

def create_asset_xml(asset_name, convex_parts=None, materials=None, submesh_files=None, module_name="object"):
    """创建资源依赖 XML 文件"""
    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")
    
    # 添加 MTL 材质
    if materials:
        for material_name, material in materials.items():
            material_elem = ET.SubElement(asset, "material")
            material_elem.set("name", f"{asset_name}_{material_name}")
            material_elem.set("rgba", material.mjcf_rgba())
            material_elem.set("specular", material.mjcf_specular())
            material_elem.set("shininess", material.mjcf_shininess())
            
            # 如果材质有纹理贴图
            if material.map_Kd:
                # 添加纹理
                texture_elem = ET.SubElement(asset, "texture")
                texture_elem.set("type", "2d")
                texture_elem.set("name", f"{asset_name}_{material_name}_texture")
                texture_elem.set("file", f"{module_name}/{asset_name}/{material.map_Kd}")
                
                # 更新材质使用纹理
                material_elem.set("texture", f"{asset_name}_{material_name}_texture")
                material_elem.attrib.pop("rgba", None)  # 移除rgba，使用纹理
    
    # 添加主网格（仅在没有材质拆分时）
    if not submesh_files:
        mesh_elem = ET.SubElement(asset, "mesh")
        mesh_elem.set("name", asset_name)
        mesh_elem.set("file", f"{module_name}/{asset_name}/{asset_name}.obj")
    
    # 添加子网格（按材质拆分的）
    if submesh_files:
        for submesh_file in submesh_files:
            submesh_name = Path(submesh_file).stem
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", submesh_name)
            part_mesh.set("file", f"{module_name}/{asset_name}/{submesh_file}")
    
    # 添加凸分解部分
    if convex_parts:
        for i in range(convex_parts):
            part_mesh = ET.SubElement(asset, "mesh")
            part_mesh.set("name", f"{asset_name}_part_{i}")
            part_mesh.set("file", f"{module_name}/{asset_name}/part_{i}.obj")
    
    return root

def create_geom_xml(asset_name, mass, diaginertia, rgba, free_joint=False,
                    convex_parts=None, materials=None, submesh_files=None, 
                    output_dir=DISCOVERSE_ASSETS_DIR, module_name="object"):
    """创建几何体定义 XML 文件"""
    root = ET.Element("mujocoinclude")
    
    # 添加自由关节
    if free_joint:
        joint_elem = ET.SubElement(root, "joint")
        joint_elem.set("type", "free")
    
    # 添加惯性
    inertial_elem = ET.SubElement(root, "inertial")
    inertial_elem.set("pos", "0 0 0")
    inertial_elem.set("mass", str(mass))
    inertial_elem.set("diaginertia", f"{diaginertia[0]} {diaginertia[1]} {diaginertia[2]}")
    
    # 处理按材质拆分的情况
    if submesh_files and materials:
        # 为每个子网格创建独立的几何体
        for i, submesh_file in enumerate(submesh_files):
            submesh_name = Path(submesh_file).stem
            geom_elem = ET.SubElement(root, "geom")
            geom_elem.set("type", "mesh")
            geom_elem.set("mesh", submesh_name)
            geom_elem.set("class", "visual")
            
            # 尝试找到对应的材质
            material_assigned = False
            
            # 读取子网格文件来确定使用的材质
            submesh_path = Path(output_dir) / "meshes" / module_name / asset_name / submesh_file
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
                    print(f"警告: 无法读取子网格文件 {submesh_path}: {e}")
            
            # 如果没有找到对应的材质，使用默认颜色，反之使用指定颜色
            if not material_assigned and rgba is not None:
                geom_elem.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    
    elif materials and len(materials) == 1:
        # 单材质情况
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("name", f"{asset_name}_visual")
        geom_elem.set("mesh", asset_name)
        geom_elem.set("class", "visual")
        
        # 使用单个材质
        material_name = list(materials.keys())[0]
        geom_elem.set("material", f"{asset_name}_{material_name}")
    
    elif convex_parts:
        # 先添加用于可视化的完整网格几何体
        visual_geom = ET.SubElement(root, "geom")
        if rgba is not None:
            visual_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        visual_geom.set("mesh", asset_name)
        visual_geom.set("class", "visual")
        
        # 然后添加凸分解的多个碰撞部分
        for i in range(convex_parts):
            collision_geom = ET.SubElement(root, "geom")
            collision_geom.set("type", "mesh")
            collision_geom.set("name", f"{asset_name}_collision_part_{i}")
            collision_geom.set("class", "collision")
            if rgba is not None:
                collision_geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
            collision_geom.set("mesh", f"{asset_name}_part_{i}")
    else:
        # 单个几何体
        geom_elem = ET.SubElement(root, "geom")
        geom_elem.set("type", "mesh")
        geom_elem.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
        geom_elem.set("mesh", asset_name)
    
    return root

def save_xml_with_formatting(root, filepath):
    """保存格式化的 XML 文件"""
    # 使用 ElementTree 的 indent 方法进行格式化（Python 3.9+）
    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8', xml_declaration=False)

def create_preview_xml(asset_name, module_name="object"):
    """创建预览用的 XML 文件"""
    root = ET.Element("mujoco")
    root.set("model", "temp_preview_env")
    
    # 选项
    option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")
    
    # 编译器
    compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "../meshes")
    compiler.set("texturedir", "../meshes/")
    
    # 包含资源
    include = ET.SubElement(root, "include")
    include.set("file", f"{module_name}/{asset_name}/{asset_name}_dependencies.xml")
    
    # 默认设置
    default = ET.SubElement(root, "default")
    obj_default = ET.SubElement(default, "default")
    obj_default.set("class", "visual")
    geom_default = ET.SubElement(obj_default, "geom")
    geom_default.set("group", "2")
    geom_default.set("type", "mesh")
    geom_default.set("contype", "0")
    geom_default.set("conaffinity", "0")
    
    # 世界体
    worldbody = ET.SubElement(root, "worldbody")
    
    # 地面
    floor_geom = ET.SubElement(worldbody, "geom")
    floor_geom.set("name", "floor")
    floor_geom.set("type", "plane")
    floor_geom.set("size", "2 2 0.1")
    floor_geom.set("rgba", ".8 .8 .8 1")
    
    # 光源
    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 3")
    light.set("dir", "0 0 -1")
    
    # 物体
    body = ET.SubElement(worldbody, "body")
    body.set("name", asset_name)
    body.set("pos", "0 0 0.5")
    
    # 包含物体定义
    body_include = ET.SubElement(body, "include")
    body_include.set("file", f"{module_name}/{asset_name}/{asset_name}.xml")
    
    return root


def process_single_file(file_info: Tuple[Path, dict]) -> Tuple[str, bool, Optional[str]]:
    """处理单个网格文件的转换。
    
    Args:
        file_info: (file_path, config) 的元组，config包含转换参数
        
    Returns:
        (asset_name, success, error_msg) 的元组
        如果成功，error_msg为None
        如果失败，error_msg包含错误信息
    """
    input_file, config = file_info
    
    try:
        rgba = config['rgba']
        mass = config['mass']
        diaginertia = config['diaginertia']
        free_joint = config['free_joint']
        convex_de = config['convex_decomposition']
        output_assets_dir = config['output']
        scene_config = config['scene']
        verbose = config.get('verbose', False)
        module_name = config.get('module', 'object')
        
        if input_file.suffix.lower() not in ['.obj', '.stl']:
            return (str(input_file), False, f"不支持的文件类型: {input_file.suffix}")

        asset_name = input_file.stem
        output_dir = os.path.join(output_assets_dir, "meshes", module_name, asset_name)
        mjcf_obj_dir = os.path.join(output_assets_dir, "mjcf", module_name, asset_name)
        
        # 创建输出目录
        if not os.path.exists(mjcf_obj_dir):
            os.makedirs(mjcf_obj_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 复制文件
        if input_file.parent != Path(output_dir):
            if input_file.suffix.lower() == ".obj":
                output_path = Path(output_dir) / input_file.name
                copy_obj_with_mtl(input_file, output_path)
            else:
                shutil.copy(input_file, output_dir)
            
        # 处理材质拆分（仅对 OBJ 文件）
        materials = {}
        submesh_files = []
        if input_file.suffix.lower() == ".obj":
            obj_path = Path(output_dir) / f"{asset_name}.obj"
            materials, submesh_files = split_obj_by_materials(obj_path, Path(output_dir))
            
            if submesh_files and verbose:
                logger.info(f"{asset_name}: 检测到多材质，已拆分为 {len(submesh_files)} 个子网格")
            
            # 处理材质的纹理文件
            if submesh_files or len(materials) == 1:
                for material_name, material in materials.items():
                    if material.map_Kd:
                        texture_src = input_file.parent / material.map_Kd
                        if texture_src.exists():
                            texture_dst = Path(output_dir) / material.map_Kd
                            shutil.copy(texture_src, texture_dst)

        convex_parts_count = 0

        # 处理凸分解
        if convex_de:
            import coacd
            import trimesh
            
            if verbose:
                logger.info(f"{asset_name}: 开始凸分解...")
            
            mesh = trimesh.load(str(input_file), force="mesh")
            mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
            coacd_config_scene = {
                "threshold": 0.01,
                "preprocess_resolution": 100,
            }
            coacd_config = coacd_config_scene if scene_config else {}
            parts = coacd.run_coacd(mesh_coacd, **coacd_config)

            for i, part in enumerate(parts):
                part_filename = f"part_{i}.obj"
                output_part_file = os.path.join(output_dir, part_filename)
                part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
                part_mesh.export(output_part_file)
            
            convex_parts_count = len(parts)
            if verbose:
                logger.info(f"{asset_name}: 凸分解完成，生成 {convex_parts_count} 个凸包")

        # 创建资源依赖 XML
        asset_xml = create_asset_xml(asset_name, convex_parts_count if convex_de else None, 
                                    materials if submesh_files or len(materials) == 1 else None, 
                                    submesh_files if submesh_files else None,
                                    module_name)
        asset_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}_dependencies.xml")
        save_xml_with_formatting(asset_xml, asset_file_path)

        # 创建几何体定义 XML
        geom_xml = create_geom_xml(asset_name, mass, diaginertia, rgba, free_joint, 
                                  convex_parts_count if convex_de else None,
                                  materials if submesh_files or len(materials) == 1 else None, 
                                  submesh_files if submesh_files else None,
                                  output_assets_dir,
                                  module_name)
        geom_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}.xml")
        save_xml_with_formatting(geom_xml, geom_file_path)

        return (asset_name, True, None)
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}" if config.get('verbose', False) else str(e)
        return (str(input_file.name), False, error_detail)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将 .obj 或 .stl 网格文件转换为 MuJoCo MJCF 格式的XML文件。")
    parser.add_argument("input_file", type=str, nargs='?', help="输入的网格文件路径 (.obj 或 .stl)。")
    parser.add_argument("-d", "--directory", type=str, help="输入目录，批量处理目录中的所有 .obj 和 .stl 文件。")
    parser.add_argument("--rgba", nargs=4, type=float, default=None, help="网格的 RGBA 颜色，不指定则不设置 rgba 属性（使用材质或默认颜色）。")
    parser.add_argument("--mass", type=float, default=0.001, help="网格的质量，默认为 0.001 kg。")
    parser.add_argument("-o", "--output", type=str, default=DISCOVERSE_ASSETS_DIR, help="输出的资产文件路径，")
    parser.add_argument("--diaginertia", nargs=3, type=float, default=[0.00002, 0.00002, 0.00002], help="网格的对角惯性张量，默认为 [2e-5, 2e-5, 2e-5]。")
    parser.add_argument("--free_joint", action="store_true", help="是否为物体添加free自由度")
    parser.add_argument("-cd", "--convex_decomposition", action="store_true", help="是否将网格分解为多个凸部分以进行更精确的碰撞检测，默认为 False。需要安装 coacd 和 trimesh。")
    parser.add_argument("--scene", action="store_true", help="进行高精度的凸分解 False。")
    parser.add_argument("--verbose", action="store_true", help="是否在转换完成后使用 MuJoCo 查看器可视化生成的模型，默认为 False。")
    parser.add_argument("--processes", type=int, default=None, help="批量处理时使用的进程数量，默认为 CPU 核心数 - 2。")
    parser.add_argument("-m", "--module", type=str, default="object", help="模块名称，用于组织资产文件，默认为 'object'")
    args = parser.parse_args()

    # 验证参数
    if not args.input_file and not args.directory:
        parser.error("必须提供 input_file 或 --directory 参数之一")
    if args.input_file and args.directory:
        parser.error("不能同时提供 input_file 和 --directory 参数")
    
    # 检查凸分解依赖
    if args.convex_decomposition:
        try:
            import coacd
            import trimesh
        except ImportError:
            logger.error("coacd 和 trimesh 未安装。请使用 'pip install coacd trimesh' 命令安装。")
            return 1

    # 准备配置
    config = {
        'rgba': args.rgba,
        'mass': args.mass,
        'diaginertia': args.diaginertia,
        'free_joint': args.free_joint,
        'convex_decomposition': args.convex_decomposition,
        'output': args.output,
        'scene': args.scene,
        'verbose': args.verbose,
        'module': args.module,
    }
    
    # 收集需要处理的文件
    mesh_files = []
    if args.directory:
        input_dir = Path(args.directory)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"目录不存在或不是有效目录: {args.directory}")
            return 1
        
        # 收集所有支持的文件（不区分大小写）
        for item in input_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ['.obj', '.stl']:
                mesh_files.append(item)
        
        if not mesh_files:
            logger.error(f"在目录 {args.directory} 中未找到任何 .obj 或 .stl 文件")
            return 1
    else:
        # 单文件模式
        input_file = Path(args.input_file)
        if not input_file.exists():
            logger.error(f"文件不存在: {input_file}")
            return 1
        
        if input_file.suffix.lower() not in ['.obj', '.stl']:
            logger.error(f"{input_file} 不是有效的文件类型。请使用 .obj 或 .stl 文件。")
            return 1
        
        mesh_files = [input_file]
    
    num_files = len(mesh_files)
    print(f"找到 {num_files} 个网格文件待处理")
    
    # 准备任务列表
    tasks = [(mesh_file, config) for mesh_file in mesh_files]
    
    # 处理文件
    results = []
    if num_files == 1:
        # 单文件处理
        print(f"处理文件: {mesh_files[0].name}")
        result = process_single_file(tasks[0])
        results.append(result)
    else:
        # 批量处理：使用多进程
        cpu_count = os.cpu_count() or 4
        if args.processes is None:
            max_processes = max(1, cpu_count - 2)
        else:
            max_processes = max(1, args.processes)
        actual_processes = min(max_processes, num_files)
        
        logger.info(f"检测到 {cpu_count} 个 CPU 核心，使用 {actual_processes} 个进程进行批量处理")
        
        if actual_processes == 1:
            # 单进程处理
            print("使用单进程模式处理...")
            for i, task in enumerate(tasks, 1):
                print(f"处理 [{i}/{num_files}]: {task[0].name}")
                result = process_single_file(task)
                results.append(result)
        else:
            # 多进程处理
            print(f"使用 {actual_processes} 个进程进行并行处理...")
            with mp.Pool(processes=actual_processes) as pool:
                results = pool.map(process_single_file, tasks)
    
    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = num_files - success_count
    
    if num_files == 1:
        # 单文件：显示详细信息
        asset_name, success, error_msg = results[0]
        if success:
            print(f"资源 {asset_name} 已成功转换为 MJCF 格式。")
            output_dir = os.path.join(args.output, "meshes", args.module, asset_name)
            mjcf_obj_dir = os.path.join(args.output, "mjcf", args.module, asset_name)
            print(f"网格文件保存在: {output_dir}")
            print(f"资源依赖文件: {os.path.join(mjcf_obj_dir, f'{asset_name}_dependencies.xml')}")
            print(f"物体定义文件: {os.path.join(mjcf_obj_dir, f'{asset_name}.xml')}")
            
            # 预览模式
            if args.verbose:
                print("\n正在启动 MuJoCo 查看器...")
                py_dir = shutil.which('python') or shutil.which('python3')
                if not py_dir:
                    logger.error("找不到 python 或 python3 可执行文件。无法启动查看器。")
                    return 1

                tmp_world_mjcf = os.path.join(args.output, "mjcf", "_tmp_preview.xml")
                preview_xml = create_preview_xml(asset_name, args.module)
                save_xml_with_formatting(preview_xml, tmp_world_mjcf)
                    
                cmd_line = f"{py_dir} -m mujoco.viewer --mjcf {tmp_world_mjcf}"
                print(f"执行命令: {cmd_line}")
                os.system(cmd_line)
        else:
            logger.error(f"转换失败 - {error_msg}")
            return 1
    else:
        # 批量处理：显示统计信息
        print(f"\n批量处理完成:")
        print(f"  成功: {success_count}/{num_files}")
        print(f"  失败: {fail_count}/{num_files}")
        
        if fail_count > 0:
            print("\n失败的文件:")
            for name, success, error_msg in results:
                if not success:
                    print(f"  - {name}: {error_msg}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
