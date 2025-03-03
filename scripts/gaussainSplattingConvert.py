import tqdm
import struct
import numpy as np
from scipy.spatial.transform import Rotation


def convert_ply_bin2ascii(input_file, output_file):
    with open(input_file, "rb") as f:
        binary_data = f.read()

    header_end = binary_data.find(b"end_header\n") + len(b"end_header\n")
    header = binary_data[:header_end].decode("utf-8")
    body = binary_data[header_end:]

    ascii_header = header.replace("binary_little_endian", "ascii")

    with open(output_file, "w") as f:
        f.write(ascii_header)

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
                vertex_data = struct.unpack_from(vertex_format, body, offset)
                f.write(" ".join(map(str, vertex_data)) + "\n")
                offset += vertex_size
        else:
            print(
                f"Error: body size {len(body)} does not match vertex count {vertex_count} * vertex size {vertex_size}"
            )


def convert_ply_ascii2bin(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    header_end = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    header = lines[: header_end + 1]
    body = lines[header_end + 1 :]

    binary_header = "".join(header).replace("ascii", "binary_little_endian")

    vertex_format = "<3f3f3f45f1f3f4f"

    with open(output_file, "wb") as f:
        f.write(binary_header.encode("utf-8"))

        for line in tqdm.tqdm(body):
            vertex_data = list(map(float, line.split()))
            binary_data = struct.pack(vertex_format, *vertex_data)
            f.write(binary_data)


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(description="Convert binary PLY to ASCII PLY.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input binary PLY file"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output ASCII PLY file",
        default=None,
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".ply", "_ascii.ply")

    convert_ply_bin2ascii(args.input_file, args.output_file)
    # convert_ply_ascii2bin(args.input_file, args.input_file.replace('.ply', '_reconverted.ply'))
