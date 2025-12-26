from typing import Union

import os
import sys
import argparse

import torch
import numpy as np
from scipy.spatial.transform import Rotation

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from discoverse.gaussian_renderer.gaussiandata import GaussianData
from discoverse.gaussian_renderer.super_splat_loader import save_super_splat_ply
from discoverse.gaussian_renderer.util_gau import load_ply, save_ply, transform_shs

def transform_mesh(
        mesh_path: Union[str, os.PathLike], 
        transformMatrix: np.ndarray, 
        scale_factor: float, 
        output_path: Union[str, os.PathLike], 
        rescale_first: bool = True, 
        silent: bool = False) -> None:
    """
    Apply transformation to a mesh and save the transformed mesh.

    Args:
        mesh_path (Union[str, os.PathLike]): Path to the input mesh file.
        transformMatrix (np.ndarray): (4, 4) transformation matrix.
        scale_factor (float): Scale factor to apply.
        output_path (Union[str, os.PathLike]): Path to save the transformed mesh.
        rescale_first (bool): Whether to rescale before transforming. Defaults to True.
        silent (bool): Whether to suppress output. Defaults to False.
    """
    assert isinstance(transformMatrix, np.ndarray) and transformMatrix.shape == (4,4)

    import trimesh
    mesh = trimesh.load(mesh_path)

    if rescale_first and scale_factor != 1.0:
        S = np.diag([scale_factor, scale_factor, scale_factor, 1.0])
        mesh.apply_transform(S)
        if not silent:
            print(f"First: Rescaled mesh with factor {scale_factor}")

    mesh.apply_transform(transformMatrix)
    if not silent:
        print("Applied transformation matrix to mesh.")

    if not rescale_first and scale_factor != 1.0:
        S = np.diag([scale_factor, scale_factor, scale_factor, 1.0])
        mesh.apply_transform(S)
        if not silent:
            print(f"Rescaled mesh with factor {scale_factor}")

    if not silent:
        print(f"Saving transformed mesh to {output_path}...")
    mesh.export(output_path)

def transform_gaussian(gaussian_data: GaussianData, transformMatrix: np.ndarray, scale_factor: float = 1., rescale_first: bool = True, silent: bool = False) -> GaussianData:
    """
    Apply transformation to Gaussian data.

    Args:
        gaussian_data (GaussianData): The Gaussian data to transform.
        transformMatrix (np.ndarray): (4, 4) transformation matrix.
        scale_factor (float): Scale factor to apply. Defaults to 1.0.
        rescale_first (bool): Whether to rescale before transforming. Defaults to True.
        silent (bool): Whether to suppress output. Defaults to False.

    Returns:
        GaussianData: The transformed Gaussian data.
    """
    assert isinstance(transformMatrix, np.ndarray) and transformMatrix.shape == (4,4)
    
    if not silent:
        print("Processing...")
    
    # 1. Transform Positions
    xyz = gaussian_data.xyz
    R = transformMatrix[:3, :3]
    t = transformMatrix[:3, 3]
    
    if rescale_first and scale_factor != 1.0:
        # Rescale first
        xyz *= scale_factor
        gaussian_data.scale *= scale_factor
        if not silent:
            print(f"First: Rescaled positions and scales with factor {scale_factor}")
    
    # xyz_new = (R @ xyz.T).T + t
    xyz_new = np.dot(xyz, R.T) + t
    
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
    if not rescale_first and scale_factor != 1.0:
        gaussian_data.xyz *= scale_factor
        gaussian_data.scale *= scale_factor
        if not silent:
            print(f"Rescaled positions and scales with factor {scale_factor}")
    
    # 4. Transform SH Features
    sh = gaussian_data.sh
    # Ensure sh is (N, K, 3)
    if len(sh.shape) == 2:
        sh = sh.reshape(sh.shape[0], -1, 3)
    
    # Check if we have higher order SH
    if sh.shape[1] > 1:
        if not silent:
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
    parser.add_argument('--sh-degree', type=int, help='SH degree to save', default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rescale-first', action='store_true', help='Rescale before transforming (default)', default=True)
    group.add_argument('--transform-first', dest='no_rescale_first', action='store_true', help='Transform before rescaling', default=False)
    args = parser.parse_args()

    Tmat = np.eye(4)
    if args.transform is not None:
        Tmat[:3,3] = args.transform
    
    if args.rotation is not None:
        Tmat[:3,:3] = Rotation.from_quat(args.rotation).as_matrix()


    rescale_first = not args.no_rescale_first

    is_gaussian_file = False

    print(f"Reading {args.input_file}...")
    if args.input_file.lower().endswith('.obj') or args.input_file.lower().endswith('.stl') or args.input_file.lower().endswith('.dae'):
        print("Input is a mesh file.")
        is_gaussian_file = False
    else:
        try:
            gaussian_data = load_ply(args.input_file)
            is_gaussian_file = True
        except Exception as e:
            print(f"Failed to load as Gaussian PLY: {e}")
            is_gaussian_file = False
            
    if is_gaussian_file:
        if args.output_file is None:
            args.output_file = args.input_file.replace('.ply', '_trans.ply')

        gaussian_data_new = transform_gaussian(gaussian_data, Tmat, scale_factor=args.scale, rescale_first=rescale_first)

        if args.compress:
            print(f"Compress and save to {args.output_file}...")
            save_super_splat_ply(gaussian_data_new, args.output_file, save_sh_degree=args.sh_degree)
        else:
            print(f"Writing to {args.output_file}...")
            save_ply(gaussian_data_new, args.output_file, save_sh_degree=args.sh_degree)
    else:
        if args.output_file is None:
            suffix = os.path.splitext(args.input_file)[-1]
            args.output_file = args.input_file.replace(suffix, '_trans' + suffix)
        transform_mesh(args.input_file, Tmat, scale_factor=args.scale, output_path=args.output_file, rescale_first=rescale_first)