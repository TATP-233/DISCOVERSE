import tqdm
import struct
import numpy as np


def read_binply_proc(input_file, output_file):
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
        vertex_data = list(struct.unpack_from(vertex_format, body, offset))
        offset += vertex_size
        xyz = np.array(vertex_data[:3])

        if 0.1 < xyz[2] < 0.85 and -1.4 < xyz[0] < 1.4 and -1.4 < xyz[1] < 1.4:
            continue
        else:
            data.append(vertex_data)

    data_arr = np.array(data)

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

    parser = argparse.ArgumentParser(
        description="Convert ASCII PLY delete position range point."
    )
    parser.add_argument("input_file", type=str, help="Path to the input ASCII PLY file")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output BIN PLY file",
        default=None,
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".ply", "_new.ply")

    read_binply_proc(args.input_file, args.output_file)
