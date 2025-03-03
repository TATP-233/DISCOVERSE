import tqdm
import struct
import numpy as np
from scipy.spatial.transform import Rotation


def convert_ply_zeroSH(input_file, output_file):
    with open(input_file, "rb") as f:
        binary_data = f.read()

    header_end = binary_data.find(b"end_header\n") + len(b"end_header\n")
    header = binary_data[:header_end].decode("utf-8")
    body = binary_data[header_end:]

    with open(output_file, "wb") as f:
        f.write(header.encode("utf-8"))

        offset = 0
        # x, y, z                    [0,2)
        # nx, ny, nz                 [2,5)
        # f_dc_0, f_dc_1, f_dc_2     [5,8)
        # f_rest_0 to f_rest_44      [9,54)
        # opacity                    [54]
        # scale_0, scale_1, scale_2  [55,58)
        # rot_0, rot_1, rot_2, rot_3 [58,63)
        vertex_format = "<3f3f3f45f1f3f4f"

        vertex_size = struct.calcsize(vertex_format)
        vertex_count = int(header.split("element vertex ")[1].split()[0])

        if len(body) == vertex_count * vertex_size:
            for _ in tqdm.trange(vertex_count):
                vertex_data = list(struct.unpack_from(vertex_format, body, offset))
                vertex_data[9:54] = [0] * 45
                binary_data = struct.pack(vertex_format, *vertex_data)
                f.write(binary_data)
                offset += vertex_size
        else:
            print(
                f"Error: body size {len(body)} does not match vertex count {vertex_count} * vertex size {vertex_size}"
            )


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description="Convert binary PLY zero SH.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input binary PLY file"
    )

    args = parser.parse_args()

    convert_ply_zeroSH(args.input_file, args.input_file.replace(".ply", "_shzero.ply"))
