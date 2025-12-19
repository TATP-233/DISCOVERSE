import argparse
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation

import torch
import einops
from einops import einsum
from e3nn import o3

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from discoverse.gaussian_renderer.util_gau import load_ply, save_ply
from discoverse.gaussian_renderer.super_splat_loader import save_super_splat_ply

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

def transform_gaussian(gaussian_data, transformMatrix, scale_factor=1., slient=False):
    assert type(transformMatrix) == np.ndarray and transformMatrix.shape == (4,4)
    
    if not slient:
        print("Processing...")
    
    # 1. Transform Positions
    xyz = gaussian_data.xyz
    R = transformMatrix[:3, :3]
    t = transformMatrix[:3, 3]
    
    # xyz_new = (R @ xyz.T).T + t
    xyz_new = np.dot(xyz, R.T) + t
    
    # Scale positions
    if scale_factor != 1.0:
        xyz_new *= scale_factor
        if not slient:
            print(f"Rescaled positions with factor {scale_factor}")
    
    gaussian_data.xyz = xyz_new.astype(np.float32)
    
    # 2. Transform Rotations
    # rot is (N, 4) wxyz
    rot = gaussian_data.rot
    # Convert to matrix
    # scipy Rotation uses xyzw
    r_orig = Rotation.from_quat(rot[:, [1, 2, 3, 0]]) 
    mat_orig = r_orig.as_matrix()
    
    # Apply rotation R
    mat_new = np.matmul(R, mat_orig)
    
    # Convert back to quat
    r_new = Rotation.from_matrix(mat_new)
    rot_new_xyzw = r_new.as_quat()
    rot_new = rot_new_xyzw[:, [3, 0, 1, 2]] # xyzw -> wxyz
    
    gaussian_data.rot = rot_new.astype(np.float32)
    
    # 3. Transform Scales
    if scale_factor != 1.0:
        gaussian_data.scale *= scale_factor
        if not slient:
            print(f"Rescaled scales with factor {scale_factor}")
    
    # 4. Transform SH Features
    sh = gaussian_data.sh
    # Ensure sh is (N, K, 3)
    if len(sh.shape) == 2:
        sh = sh.reshape(sh.shape[0], -1, 3)
    
    # Check if we have higher order SH
    if sh.shape[1] > 1:
        if not slient:
            print("Processing SH features...")
        # DC is sh[:, 0, :]
        # Rest is sh[:, 1:, :]
        sh_rest = sh[:, 1:, :] # (N, K-1, 3)
        
        sh_rest_tensor = torch.from_numpy(sh_rest).float()
        
        # transform_shs modifies in place or returns new tensor
        sh_rest_transformed = transform_shs(sh_rest_tensor, R)
        
        sh[:, 1:, :] = sh_rest_transformed.numpy()
        gaussian_data.sh = sh
    
    return gaussian_data



if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description='example: python3 scripts/gsply_edit.py -i data/ply/000000.ply -o data/ply/000000_trans.ply -t [0, 0, 0] -r [0.707, 0., 0., 0.707] -s 1')
    parser.add_argument('input_file', type=str, help='Path to the input binary PLY file')
    parser.add_argument('-c', '--compress', action='store_true', help='Save as compressed PLY', default=False)
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

    print(f"Reading {args.input_file}...")
    gaussian_data = load_ply(args.input_file)

    gaussian_data_new = transform_gaussian(gaussian_data, Tmat, scale_factor=args.scale)

    if args.compress:
        print(f"Compress and save to {args.output_file}...")
        save_super_splat_ply(gaussian_data_new, args.output_file)
    else:
        print(f"Writing to {args.output_file}...")
        save_ply(gaussian_data_new, args.output_file)