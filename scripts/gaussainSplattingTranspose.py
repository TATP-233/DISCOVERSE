import tqdm
import struct
import numpy as np
from scipy.spatial.transform import Rotation

import torch
import einops
from einops import einsum
from e3nn import o3


def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # switch axes: yzx -> xyz
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(
        torch.from_numpy(permuted_rotation_matrix)
    )

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, "n shs_num rgb -> n rgb shs_num")
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, "n shs_num rgb -> n rgb shs_num")
    two_degree_shs = einsum(
        D_2,
        two_degree_shs,
        "... i j, ... j -> ... i",
    )
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(
        three_degree_shs, "n shs_num rgb -> n rgb shs_num"
    )
    three_degree_shs = einsum(
        D_3,
        three_degree_shs,
        "... i j, ... j -> ... i",
    )
    three_degree_shs = einops.rearrange(
        three_degree_shs, "n rgb shs_num -> n shs_num rgb"
    )
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def rescale(xyz, scales, scale: float):
    if scale != 1.0:
        xyz *= scale
        scales += np.log(scale)
        print("rescaled with factor {}".format(scale))
    return xyz, scales


def ply_bin_transpose(input_file, output_file, transformMatrix, scale_factor=1.0):
    assert type(transformMatrix) == np.ndarray and transformMatrix.shape == (4, 4)

    with open(input_file, "rb") as f:
        binary_data = f.read()

    header_end = binary_data.find(b"end_header\n") + len(b"end_header\n")
    header = binary_data[:header_end].decode("utf-8")
    body = binary_data[header_end:]

    offset = 0
    vertex_format = "<3f3f3f45f1f3f4f"

    vertex_size = struct.calcsize(vertex_format)
    vertex_count = int(header.split("element vertex ")[1].split()[0])

    if len(body) != vertex_count * vertex_size:
        print(
            f"Error: body size {len(body)} does not match vertex count {vertex_count} * vertex size {vertex_size}"
        )
        return

    data = []
    for _ in tqdm.trange(vertex_count):
        vertex_data = struct.unpack_from(vertex_format, body, offset)
        data.append(vertex_data)
        offset += vertex_size
    data_arr = np.array(data)

    pose_arr = np.zeros((data_arr.shape[0], 4, 4))
    pose_arr[:, 3, 3] = 1
    pose_arr[:, :3, 3] = data_arr[:, :3]
    quat_wxyz = data_arr[:, -4:]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    pose_arr[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()

    trans_pose_arr = transformMatrix @ pose_arr[:]

    data_arr[:, :3] = trans_pose_arr[:, :3, 3]
    quat_arr = Rotation.from_matrix(trans_pose_arr[:, :3, :3]).as_quat()
    data_arr[:, -4:] = quat_arr[:, [3, 0, 1, 2]]

    RMat = transformMatrix[:3, :3]

    f_rest = torch.from_numpy(data_arr[:, 9:54].reshape((-1, 3, 15)).transpose(0, 2, 1))
    shs = transform_shs(f_rest, RMat).numpy()
    shs = shs.transpose(0, 2, 1).reshape(-1, 45)
    data_arr[:, 9:54] = shs

    xyz, scales = rescale(data_arr[:, :3], data_arr[:, 55:58], scale_factor)
    data_arr[:, :3] = xyz
    data_arr[:, 55:58] = scales

    offset = 0
    with open(output_file, "wb") as f:
        f.write(
            header.replace(f"{vertex_count}", f"{data_arr.shape[0]}").encode("utf-8")
        )

        for vertex_data in tqdm.tqdm(data_arr):
            binary_data = struct.pack(vertex_format, *(vertex_data.tolist()))
            f.write(binary_data)


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description="Transpose binary PLY.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input binary PLY file"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output PLY file",
        default=None,
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".ply", "_trans.ply")

    Tmat = np.eye(4)
    Tmat[2, 3] = -0.01
    ply_bin_transpose(args.input_file, args.output_file, Tmat, scale_factor=1)
