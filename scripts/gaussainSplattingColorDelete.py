import tqdm
import struct
import colorsys
import numpy as np

C0 = 0.28209479177387814


def SH2RGB(sh):
    return np.clip(sh * C0 + 0.5, 0.0, 1.0)


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
        f_dc = np.array(vertex_data[6:9])
        rgb = SH2RGB(f_dc)
        hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

        # if (abs(hsv[0] - 0.333) < 0.1 or hsv[0] > 5/6 or hsv[0] < 1/6) and hsv[1] > 0.5:
        # if (hsv[0] > 11/12 or hsv[0] < 1/12) and hsv[1] > 0.9:
        if abs(hsv[0] - 0.333) < 0.15 and hsv[1] > 0.0:  # 删除绿色
            # if abs(hsv[0] - 0.333) < 0.01: # 删除绿色
            # if hsv[2] < 0.7: # 删除黑色
            pass
            # continue

        # if hsv[1] < 0.1 and hsv[2] > 0.9:
        #     continue

        # print(np.array(hsv), np.array(rgb), np.array(f_dc))
        xyz = np.array(vertex_data[:3])
        # if 1 or np.linalg.norm(xyz) < 1.0:
        # if hsv[2] < 0.1 and np.linalg.norm(xyz) < 1.0: # 保留黑色
        if hsv[1] < 0.5 and np.linalg.norm(xyz) < 1.0:  # 保留黑色和白色
            vertex_data[9:54] = [0] * 45
            data.append(vertex_data)

    data_arr = np.array(data)

    with open(output_file, "wb") as f:
        f.write(
            header.replace(f"{vertex_count}", f"{data_arr.shape[0]}").encode("utf-8")
        )

        for vertex_data in tqdm.tqdm(data_arr):
            binary_data = struct.pack(vertex_format, *(vertex_data.tolist()))
            f.write(binary_data)


def read_asciiply_proc(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    header_end = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    header = lines[: header_end + 1]
    body = lines[header_end + 1 :]
    points_cnt = 0

    # binary_header = ''.join(header).replace('ascii', 'binary_little_endian')
    # vertex_format = '<3f3f3f45f1f3f4f'

    # with open(output_file, 'wb') as f:
    #     f.write(binary_header.encode('utf-8'))
    #     for line in tqdm.tqdm(body):
    #         vertex_data = list(map(float, line.split()))
    #         f_dc = vertex_data[6:9]
    #         if f_dc[1] > f_dc[0] + f_dc[2] and f_dc[1] > 0.0:
    #             continue
    #         else:
    #             binary_data = struct.pack(vertex_format, *vertex_data)
    #             f.write(binary_data)
    #             points_cnt += 1

    with open(output_file, "w") as f:
        for line in header:
            f.write(line)
        for line in tqdm.tqdm(body):
            vertex_data = list(map(float, line.split()))
            f_dc = vertex_data[6:9]
            if f_dc[1] > f_dc[0] + f_dc[2] and f_dc[1] > 0.0:
                continue
            else:
                f.write(" ".join(map(str, vertex_data)) + "\n")
                points_cnt += 1
    print(f"Total points: {points_cnt}")


if __name__ == "__main__":
    import argparse

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(
        description="Convert ASCII PLY delete color point."
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

    # read_asciiply_proc(args.input_file, args.output_file)
    read_binply_proc(args.input_file, args.output_file)
