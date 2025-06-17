import os
import shutil
import argparse
from discoverse import DISCOVERSE_ASSETS_DIR

# --- 使用示例 ---
"""
命令行使用示例:

1. 基本转换 (使用默认颜色、质量、惯性，不固定，不分解):
    python scripts/mesh2mjcf.py /path/to/your/model.obj

2. 指定 RGBA 颜色:
    python scripts/mesh2mjcf.py /path/to/your/model.stl --rgba 0.8 0.2 0.2 1.0

3. 使用纹理 (需要提供纹理文件路径):
    python scripts/mesh2mjcf.py /path/to/your/model.obj --texture /path/to/your/texture.png

4. 固定物体 (不能自由移动):
    python scripts/mesh2mjcf.py /path/to/your/model.obj --fix_joint

5. 进行凸分解 (用于更精确碰撞):
    python scripts/mesh2mjcf.py /path/to/your/model.obj -cd

6. 转换后立即用 MuJoCo 查看器预览:
    python scripts/mesh2mjcf.py /path/to/your/model.obj --verbose

7. 组合使用 (例如，使用纹理、进行凸分解并预览):
    python scripts/mesh2mjcf.py /path/to/your/model.obj --texture /path/to/texture.png -cd --verbose

8. 指定质量和惯性:
    python scripts/mesh2mjcf.py /path/to/your/model.obj --mass 0.5 --diaginertia 0.01 0.01 0.005
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 .obj 或 .stl 网格文件转换为 MuJoCo MJCF 格式的XML文件。")
    parser.add_argument("input_file", type=str, help="输入的网格文件路径 (.obj 或 .stl)，请使用完整路径。")
    parser.add_argument("--rgba", nargs=4, type=float, default=[0.5, 0.5, 0.5, 1], help="网格的 RGBA 颜色，默认为 [0.5, 0.5, 0.5, 1]。")
    parser.add_argument("--texture", type=str, default=None, help="纹理文件路径 (.png)，请使用完整路径。如果提供，RGBA的透明度将被忽略。")
    parser.add_argument("--mass", type=float, default=0.001, help="网格的质量，默认为 0.001 kg。")
    parser.add_argument("--diaginertia", nargs=3, type=float, default=[0.00002, 0.00002, 0.00002], help="网格的对角惯性张量，默认为 [2e-5, 2e-5, 2e-5]。")
    parser.add_argument("--fix_joint", action="store_true", help="是否将物体的基座固定在世界坐标系中（不使用 'free' 关节），默认为 False。")
    parser.add_argument("-cd", "--convex_decomposition", action="store_true", help="是否将网格分解为多个凸部分以进行更精确的碰撞检测，默认为 False。需要安装 coacd 和 trimesh。")
    parser.add_argument("--verbose", action="store_true", help="是否在转换完成后使用 MuJoCo 查看器可视化生成的模型，默认为 False。")
    args = parser.parse_args()

    verbose = args.verbose
    convex_de = args.convex_decomposition

    if convex_de:
        try:
            import coacd
            import trimesh
        except ImportError:
            print("错误: coacd 和 trimesh 未安装。请使用 'pip install coacd trimesh' 命令安装。")
            exit(1)

    input_file = args.input_file
    rgba = args.rgba
    mass = args.mass
    diaginertia = args.diaginertia
    fix_joint = args.fix_joint

    if input_file.endswith(".obj"):
        asset_name = os.path.basename(input_file).replace(".obj", "")
    elif input_file.endswith(".stl"):
        asset_name = os.path.basename(input_file).replace(".stl", "")
    else:
        exit(f"错误: {input_file} 不是有效的文件类型。请使用 .obj 或 .stl 文件。")

    texture_name = None
    if args.texture is not None and os.path.exists(args.texture):
        texture_name = os.path.basename(args.texture)
        texture_dir = os.path.join("obj", asset_name, texture_name)
        texture_target_path = os.path.join(DISCOVERSE_ASSETS_DIR, "textures", texture_dir)
        if not os.path.exists(os.path.dirname(texture_target_path)):
            os.makedirs(os.path.dirname(texture_target_path))
        shutil.copy(args.texture, texture_target_path)
        rgba[3] = 0.0
    else:
        texture_name = None
    
    output_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "meshes", "obj", asset_name)    
    mjcf_obj_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "object")
    if not os.path.exists(mjcf_obj_dir):
        os.makedirs(mjcf_obj_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if os.path.dirname(input_file) != output_dir:
        shutil.copy(input_file, output_dir)
    asset_config = """<mujocoinclude>\n  <asset>\n"""

    geom_config = f"""<mujocoinclude>\n"""
    if not fix_joint:
        geom_config += f'  <joint type="free"/>\n'
    geom_config += f'  <inertial pos="0 0 0" mass="{mass}" diaginertia="{diaginertia[0]} {diaginertia[1]} {diaginertia[2]}" />\n'

    if texture_name is not None:
        asset_config += f'    <texture type="2d" name="{asset_name}_texture" file="{texture_dir}"/>\n'
        asset_config += f'    <material name="{asset_name}_texture" texture="{asset_name}_texture"/>\n\n'
        geom_config += f'  <geom material="{asset_name}_texture" mesh="{asset_name}" class="obj_visual"/>\n\n'

    asset_config += '    <mesh name="{}" file="obj/{}/{}.obj"/>\n'.format(asset_name, asset_name, os.path.basename(asset_name))

    if convex_de:
        print(f"正在对 {asset_name} 进行凸分解...")
        mesh = trimesh.load(input_file, force="mesh")
        mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh_coacd)

        for i, part in enumerate(parts):
            part_filename = f"part_{i}.obj"
            output_part_file = os.path.join(output_dir, part_filename)
            part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
            part_mesh.export(output_part_file)
            asset_config += '    <mesh name="{}_part_{}" file="obj/{}/{}"/>\n'.format(asset_name, i, asset_name, part_filename)
            geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}_part_{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name, i)
        print(f"资源 {asset_name} 已被分解为 {len(parts)} 个凸包部分。")
        
        if texture_name is None:
            pass
    else:
        geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name)

    asset_config += '  </asset>\n</mujocoinclude>\n'
    asset_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}_dependencies.xml")
    with open(asset_file_path, "w") as f:
        f.write(asset_config)

    geom_config += '</mujocoinclude>\n'
    geom_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}.xml")
    with open(geom_file_path, "w") as f:
        f.write(geom_config)

    print(f"资源 {asset_name} 已成功转换为 MJCF 格式。")
    print(f"网格文件保存在: {output_dir}")
    print(f"资源依赖文件: {asset_file_path}")
    print(f"物体定义文件: {geom_file_path}")

    if verbose:
        print("\n正在启动 MuJoCo 查看器...")
        py_dir = shutil.which('python3')
        if not py_dir:
             py_dir = shutil.which('python')
        if not py_dir:
            print("错误：找不到 python 或 python3 可执行文件。无法启动查看器。")
            exit(1)

        tmp_world_mjcf = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "_tmp_preview.xml")

        tmp_content = ""
        tmp_content += '<mujoco model="temp_preview_env">\n'
        tmp_content += '  <option gravity="0 0 -9.81"/>\n'
        tmp_content += '  <compiler meshdir="../meshes" texturedir="../textures/"/>\n'
        tmp_content +=f'  <include file="object/{asset_name}_dependencies.xml"/>\n'
        tmp_content += '  <default>\n'
        tmp_content += '    <default class="obj_visual">\n'
        tmp_content += '      <geom group="2" type="mesh" contype="0" conaffinity="0"/>\n'
        tmp_content += '    </default>\n'
        tmp_content += '  </default>\n'
        tmp_content += '  <worldbody>\n'
        tmp_content += '    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>\n'
        tmp_content += '    <light pos="0 0 3" dir="0 0 -1"/>\n'
        tmp_content +=f'    <body name="{asset_name}" pos="0 0 0.5">\n'
        tmp_content +=f'      <include file="object/{asset_name}.xml"/>\n'
        tmp_content += '    </body>\n'
        tmp_content += '  </worldbody>\n'
        tmp_content += '</mujoco>\n'

        with open(tmp_world_mjcf, "w") as f:
            f.write(tmp_content)
            
        cmd_line = f"{py_dir} -m mujoco.viewer --mjcf={tmp_world_mjcf}"
        print(f"执行命令: {cmd_line}")
        os.system(cmd_line)
        
        # print(f"删除临时预览文件: {tmp_world_mjcf}")
        # os.remove(tmp_world_mjcf)