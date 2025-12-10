import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

import torch
import einops
from einops import einsum
from e3nn import o3

def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix).float())
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] < 8:
        return shs_feat    
    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs
    if shs_feat.shape[1] < 15:
        return shs_feat

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat

def rescale(xyz, scales, scale: float):
    if scale != 1.:
        xyz *= scale
        scales += np.log(scale)
        print("rescaled with factor {}".format(scale))
    return xyz, scales

def ply_bin_transpose(input_file, output_file, transformMatrix, scale_factor=1.):
    assert type(transformMatrix) == np.ndarray and transformMatrix.shape == (4,4)

    print(f"Reading {input_file}...")
    plydata = PlyData.read(input_file)
    vertex = plydata['vertex']
    
    # Extract positions
    print("Processing positions...")
    x = np.asarray(vertex['x'])
    y = np.asarray(vertex['y'])
    z = np.asarray(vertex['z'])
    xyz = np.stack([x, y, z], axis=-1)
    
    # Extract rotations
    print("Processing rotations...")
    rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    rots = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=-1)
    
    # 3DGS stores rotation as quaternion w, x, y, z
    quat_wxyz = rots
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    
    # Construct poses
    N = xyz.shape[0]
    pose_arr = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    pose_arr[:, :3, 3] = xyz
    
    # Convert quaternions to rotation matrices
    r = Rotation.from_quat(quat_xyzw)
    pose_arr[:, :3, :3] = r.as_matrix()
    
    # Apply transformation
    trans_pose_arr = transformMatrix @ pose_arr
    
    # Extract new positions
    xyz_new = trans_pose_arr[:, :3, 3]
    
    # Extract new rotations
    r_new = Rotation.from_matrix(trans_pose_arr[:, :3, :3])
    quat_xyzw_new = r_new.as_quat()
    # Convert back to wxyz
    quat_wxyz_new = quat_xyzw_new[:, [3, 0, 1, 2]]
    
    # Update vertex data
    vertex['x'] = xyz_new[:, 0]
    vertex['y'] = xyz_new[:, 1]
    vertex['z'] = xyz_new[:, 2]
    
    for i, n in enumerate(rot_names):
        vertex[n] = quat_wxyz_new[:, i]
        
    # Extract scales
    print("Processing scales...")
    scale_names = ['scale_0', 'scale_1', 'scale_2']
    scales = np.stack([np.asarray(vertex[n]) for n in scale_names], axis=-1)
    
    # Rescale
    xyz_new, scales_new = rescale(xyz_new, scales, scale_factor)
    
    # Update scales and positions (again, because rescale modifies xyz)
    vertex['x'] = xyz_new[:, 0]
    vertex['y'] = xyz_new[:, 1]
    vertex['z'] = xyz_new[:, 2]
    
    for i, n in enumerate(scale_names):
        vertex[n] = scales_new[:, i]
        
    # SH Features
    print("Processing SH features...")
    prop_names = [p.name for p in vertex.properties]
    f_rest_names = [n for n in prop_names if n.startswith('f_rest_')]
    f_rest_names.sort(key=lambda x: int(x.split('_')[-1]))
    
    if len(f_rest_names) > 0:
        f_rest = np.stack([np.asarray(vertex[n]) for n in f_rest_names], axis=-1)
        
        sh_dc_num = 3 # Assuming RGB
        sh_rest_num = len(f_rest_names)
        
        # Reshape to (N, 3, 15) -> (N, 15, 3)
        f_rest_tensor = torch.from_numpy(f_rest.reshape((-1, sh_dc_num, sh_rest_num//sh_dc_num)).transpose(0,2,1)).float()
        
        RMat = transformMatrix[:3, :3]
        shs = transform_shs(f_rest_tensor, RMat).numpy()
        
        # Reshape back
        shs = shs.transpose(0,2,1).reshape(-1, sh_rest_num)
        
        for i, n in enumerate(f_rest_names):
            vertex[n] = shs[:, i]
            
    print(f"Writing to {output_file}...")
    plydata.write(output_file)


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description='example: python3 scripts/ply_transpose.py -i data/ply/000000.ply -o data/ply/000000_trans.ply -t [0, 0, 0] -r [0.707, 0., 0., 0.707] -s 1')
    parser.add_argument('input_file', type=str, help='Path to the input binary PLY file')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output PLY file', default=None)
    parser.add_argument('-t', '--transform', nargs=3, type=float, help='transformation', default=None)
    parser.add_argument('-r', '--rotation', nargs=4, type=float, help='rotation quaternion xyzw', default=None)
    parser.add_argument('-s', '--scale', type=float, help='Scale factor', default=1.0)
    args = parser.parse_args()

    Tmat = np.eye(4)
    if args.transform is not None:
        Tmat[:3,3] = args.transform
    
    if args.rotation is not None:
        Tmat[:3,:3] = Rotation.from_quat(args.rotation).as_matrix()

    if args.output_file is None:
        args.output_file = args.input_file.replace('.ply', '_trans.ply')

    ply_bin_transpose(args.input_file, args.output_file, Tmat, scale_factor=args.scale)
