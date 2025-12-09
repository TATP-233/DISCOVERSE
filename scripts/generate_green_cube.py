import numpy as np
import os

def generate_green_cube(filename):
    # Cube dimensions
    width = 0.03
    depth = 0.03
    height = 0.04
    
    num_points = 10000
    
    # 1. Position (x, y, z)
    # Center at (0,0,0)
    x = (np.random.rand(num_points) - 0.5) * width
    y = (np.random.rand(num_points) - 0.5) * depth
    z = (np.random.rand(num_points) - 0.5) * height
    
    # 2. Normals (nx, ny, nz) - Optional but present in example
    nx = np.zeros(num_points)
    ny = np.zeros(num_points)
    nz = np.zeros(num_points)
    
    # 3. Color (SH DC)
    # Green: RGB(0, 1, 0)
    # SH_DC = (RGB - 0.5) / 0.28209479177387814
    C0 = 0.28209479177387814
    r, g, b = 0.0, 1.0, 0.0
    f_dc_0 = np.full(num_points, (r - 0.5) / C0)
    f_dc_1 = np.full(num_points, (g - 0.5) / C0)
    f_dc_2 = np.full(num_points, (b - 0.5) / C0)
    
    # 4. SH Rest (f_rest_0 to f_rest_44)
    # Set to 0 for view-independent color
    f_rest = np.zeros((num_points, 45))
    
    # 5. Opacity
    # Inverse sigmoid: logit(opacity)
    # Want high opacity, e.g., 0.99
    # logit(p) = ln(p/(1-p))
    op_val = 0.99
    logit_op = np.log(op_val / (1 - op_val))
    opacity = np.full(num_points, logit_op)
    
    # 6. Scale
    # Log scale
    # Scale size approx 0.001
    scale_val = 0.001
    log_scale = np.log(scale_val)
    scale_0 = np.full(num_points, log_scale)
    scale_1 = np.full(num_points, log_scale)
    scale_2 = np.full(num_points, log_scale)
    
    # 7. Rotation (Quaternion w, x, y, z) -> rot_0, rot_1, rot_2, rot_3
    # Identity: (1, 0, 0, 0)
    rot_0 = np.ones(num_points)
    rot_1 = np.zeros(num_points)
    rot_2 = np.zeros(num_points)
    rot_3 = np.zeros(num_points)
    
    # Construct structured array for PLY
    # Define dtype
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'f4'))
        
    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])
    
    vertex_data = np.zeros(num_points, dtype=dtype_list)
    
    vertex_data['x'] = x.astype(np.float32)
    vertex_data['y'] = y.astype(np.float32)
    vertex_data['z'] = z.astype(np.float32)
    vertex_data['nx'] = nx.astype(np.float32)
    vertex_data['ny'] = ny.astype(np.float32)
    vertex_data['nz'] = nz.astype(np.float32)
    vertex_data['f_dc_0'] = f_dc_0.astype(np.float32)
    vertex_data['f_dc_1'] = f_dc_1.astype(np.float32)
    vertex_data['f_dc_2'] = f_dc_2.astype(np.float32)
    
    for i in range(45):
        vertex_data[f'f_rest_{i}'] = f_rest[:, i].astype(np.float32)
        
    vertex_data['opacity'] = opacity.astype(np.float32)
    vertex_data['scale_0'] = scale_0.astype(np.float32)
    vertex_data['scale_1'] = scale_1.astype(np.float32)
    vertex_data['scale_2'] = scale_2.astype(np.float32)
    vertex_data['rot_0'] = rot_0.astype(np.float32)
    vertex_data['rot_1'] = rot_1.astype(np.float32)
    vertex_data['rot_2'] = rot_2.astype(np.float32)
    vertex_data['rot_3'] = rot_3.astype(np.float32)
    
    # Write PLY file
    with open(filename, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode('ascii'))
        
        properties = [
            "x", "y", "z",
            "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2"
        ]
        properties.extend([f"f_rest_{i}" for i in range(45)])
        properties.extend([
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3"
        ])
        
        for prop in properties:
            f.write(f"property float {prop}\n".encode('ascii'))
            
        f.write(b"end_header\n")
        
        # Binary data
        f.write(vertex_data.tobytes())
        
    print(f"Generated {filename} with {num_points} points.")

if __name__ == "__main__":
      = "assets/3dgs/green_cube"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "green_cube.ply")
    generate_green_cube(output_file)
