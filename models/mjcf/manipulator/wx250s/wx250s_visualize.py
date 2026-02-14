#!/usr/bin/env python3
"""
WX250s robot arm visualization tool.

Generate point cloud (PLY) and rendered image (PNG) from joint angles.

Usage:
    # Default pose (all zeros, gripper half open)
    python wx250s_visualize.py

    # Custom joint angles + gripper
    python wx250s_visualize.py --joints 0.08 0.048 0.032 0.0 0.0015 1.569 --gripper 0.5

    # Higher density point cloud, fully open gripper
    python wx250s_visualize.py --joints 0 -0.5 0.8 0 0 1.57 --gripper 1.0 --density 100

    # Only render image (no point cloud)
    python wx250s_visualize.py --no-pointcloud

Arguments:
    --joints   : 6 joint angles in radians [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
    --gripper  : gripper open ratio, 0.0=closed, 1.0=fully open (default: 0.5)
    --density  : point cloud sampling density in pts/cm^2 (default: 50)
    --output   : output directory (default: same as this script)
    --prefix   : output filename prefix (default: "wx250s")
    --no-pointcloud : skip point cloud generation
    --no-render     : skip image rendering
    --show     : launch interactive MuJoCo viewer
"""

import argparse
import os

import numpy as np
import mujoco

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "wx250s.xml")

ARM_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
FINGER_LEFT = "left_finger"          # prismatic [0.015, 0.037]
FINGER_RIGHT = "right_finger"        # prismatic [-0.037, -0.015]
FINGER_RANGE = (0.015, 0.037)        # left finger range (meters)

# Bodies to skip in point cloud (internal mechanism)
SKIP_BODIES = {"gripper_prop_link"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="WX250s robot arm visualization: generate point cloud and rendered image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--joints", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
        metavar=("WAIST", "SHOULDER", "ELBOW", "FOREARM_R", "WRIST_A", "WRIST_R"),
        help="6 joint angles in radians: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate",
    )
    parser.add_argument(
        "--gripper", type=float, default=0.5,
        help="Gripper open ratio: 0.0=closed, 1.0=fully open (default: 0.5)",
    )
    parser.add_argument(
        "--density", type=float, default=50,
        help="Point cloud sampling density in pts/cm^2 (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=SCRIPT_DIR,
        help="Output directory (default: script directory)",
    )
    parser.add_argument(
        "--prefix", type=str, default="wx250s",
        help="Output filename prefix (default: wx250s)",
    )
    parser.add_argument(
        "--no-pointcloud", action="store_true",
        help="Skip point cloud generation",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip image rendering",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Launch interactive MuJoCo viewer",
    )
    return parser.parse_args()


# --- Model setup ---

def load_and_pose(joints, gripper_open):
    """Load model and set to desired pose.

    Args:
        joints: list of 6 floats [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]
        gripper_open: float 0.0 (closed) to 1.0 (fully open)

    Returns:
        model, data (after forward kinematics)
    """
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Set 6 arm joints
    for name, angle in zip(ARM_JOINT_NAMES, joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle

    # Set finger positions from open ratio
    gripper_open = np.clip(gripper_open, 0.0, 1.0)
    finger_pos = FINGER_RANGE[0] + gripper_open * (FINGER_RANGE[1] - FINGER_RANGE[0])

    lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FINGER_LEFT)
    rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FINGER_RIGHT)
    data.qpos[model.jnt_qposadr[lf_id]] = finger_pos
    data.qpos[model.jnt_qposadr[rf_id]] = -finger_pos

    mujoco.mj_forward(model, data)
    return model, data


# --- Point cloud from mesh ---

def sample_triangles(vertices, faces, num_points):
    """Uniformly sample points on triangle mesh surface."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area < 1e-12:
        return np.zeros((0, 3)), np.zeros((0, 3))

    probs = areas / total_area
    tri_idx = np.random.choice(len(faces), size=num_points, p=probs)

    # Random barycentric coordinates
    r1, r2 = np.random.rand(num_points), np.random.rand(num_points)
    s = np.sqrt(r1)
    u, v, w = 1.0 - s, s * (1.0 - r2), s * r2

    points = u[:, None] * v0[tri_idx] + v[:, None] * v1[tri_idx] + w[:, None] * v2[tri_idx]

    # Surface normals
    normals = cross[tri_idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, where=norms > 1e-8, out=np.zeros_like(normals))

    return points, normals


def generate_pointcloud(model, data, density_per_cm2=50, skip_bodies=None):
    """Sample point cloud from mesh surfaces in world frame.

    Args:
        model, data: MuJoCo model/data after FK
        density_per_cm2: sampling density (points per cm^2)
        skip_bodies: set of body names to exclude

    Returns:
        points (N,3), normals (N,3), body_ids (N,)
    """
    if skip_bodies is None:
        skip_bodies = set()

    pts_per_m2 = density_per_cm2 * 1e4
    all_pts, all_nrm, all_bid = [], [], []

    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname in skip_bodies:
            continue

        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn].copy()
        faces = model.mesh_face[fa:fa + fn].copy()

        # Transform to world frame
        xpos = data.geom_xpos[gid]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        verts_w = (xmat @ verts.T).T + xpos

        # Compute surface area
        v0, v1, v2 = verts_w[faces[:, 0]], verts_w[faces[:, 1]], verts_w[faces[:, 2]]
        area = 0.5 * np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1))
        n_samples = max(50, int(area * pts_per_m2))

        pts, nrm = sample_triangles(verts_w, faces, n_samples)
        all_pts.append(pts)
        all_nrm.append(nrm)
        all_bid.append(np.full(len(pts), bid, dtype=np.int32))

    return np.concatenate(all_pts), np.concatenate(all_nrm), np.concatenate(all_bid)


def save_ply(path, points, normals=None):
    """Save point cloud as PLY file."""
    try:
        import trimesh
        cloud = trimesh.PointCloud(points)
        cloud.export(path)
    except ImportError:
        # Fallback: write PLY manually
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if normals is not None:
                f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("end_header\n")
            for i in range(len(points)):
                line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
                if normals is not None:
                    line += f" {normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f}"
                f.write(line + "\n")


# --- Rendering ---

def render_views(model, data, width=800, height=600):
    """Render multiple views of the robot."""
    renderer = mujoco.Renderer(model, height, width)
    images = []

    # Auto-compute good lookat from robot bounding box
    pts = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            pts.append(data.geom_xpos[gid])
    center = np.mean(pts, axis=0) if pts else np.array([0.2, 0, 0.2])

    views = [
        {"azimuth": 135, "elevation": -25, "label": "Front-Right"},
        {"azimuth": 225, "elevation": -25, "label": "Front-Left"},
        {"azimuth": 180, "elevation": -15, "label": "Front"},
        {"azimuth": 135, "elevation": -70, "label": "Top-Down"},
    ]

    for v in views:
        camera = mujoco.MjvCamera()
        camera.azimuth = v["azimuth"]
        camera.elevation = v["elevation"]
        camera.distance = 0.8
        camera.lookat[:] = center
        renderer.update_scene(data, camera)
        img = renderer.render().copy()
        images.append((img, v["label"]))

    renderer.close()
    return images


def save_render(path, images, title=""):
    """Save rendered views as a single image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor="white")
    if n == 1:
        axes = [axes]
    for ax, (img, label) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def save_pointcloud_image(path, points, body_ids, title=""):
    """Save point cloud visualization as image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Part coloring
    unique_bodies = np.unique(body_ids)
    cmap = plt.get_cmap("tab10", max(len(unique_bodies), 1))
    body_colors = {bid: cmap(i)[:3] for i, bid in enumerate(unique_bodies)}
    colors = np.array([body_colors[b] for b in body_ids])

    def project(ax, pts, cols, az, el, label):
        a, e = np.radians(az), np.radians(el)
        fwd = np.array([np.cos(e)*np.sin(a), -np.cos(e)*np.cos(a), np.sin(e)])
        right = np.cross(fwd, [0, 0, 1])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, fwd)
        u, v, d = pts @ right, pts @ up, pts @ fwd
        order = np.argsort(d)
        ax.scatter(u[order], v[order], c=cols[order], s=0.3,
                   alpha=0.9, edgecolors="none", rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor="white")
    views = [(135, -20, "Front-Right"), (225, -20, "Front-Left"),
             (180, -15, "Front"), (135, -70, "Top-Down")]
    for ax, (az, el, label) in zip(axes, views):
        project(ax, points, colors, az, el, label)

    if title:
        plt.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# --- Interactive viewer ---

def launch_viewer(model, data):
    """Launch interactive MuJoCo viewer."""
    import mujoco.viewer

    pts = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            pts.append(data.geom_xpos[gid])
    center = np.mean(pts, axis=0) if pts else np.array([0.2, 0, 0.2])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.8
        viewer.cam.lookat[:] = center
        print("MuJoCo viewer launched. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


# --- Main ---

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    joints = args.joints
    gripper = args.gripper

    print(f"Joint angles (rad): waist={joints[0]:.4f}  shoulder={joints[1]:.4f}  "
          f"elbow={joints[2]:.4f}  forearm_roll={joints[3]:.4f}  "
          f"wrist_angle={joints[4]:.4f}  wrist_rotate={joints[5]:.4f}")
    print(f"Gripper open: {gripper:.2f} "
          f"(finger={FINGER_RANGE[0] + gripper * (FINGER_RANGE[1] - FINGER_RANGE[0]):.4f} m)")

    model, data = load_and_pose(joints, gripper)

    # End-effector position
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_gripper")
    if ee_id >= 0:
        ee = data.site_xpos[ee_id]
        print(f"End-effector: [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}] m")

    # Point cloud
    if not args.no_pointcloud:
        print(f"\nGenerating point cloud (density={args.density:.0f} pts/cm^2)...")
        points, normals, body_ids = generate_pointcloud(
            model, data, density_per_cm2=args.density, skip_bodies=SKIP_BODIES
        )
        print(f"  Total: {len(points):,} points")

        ply_path = os.path.join(args.output, f"{args.prefix}_pointcloud.ply")
        save_ply(ply_path, points, normals)
        print(f"  PLY: {ply_path}")

        pc_img_path = os.path.join(args.output, f"{args.prefix}_pointcloud.png")
        save_pointcloud_image(
            pc_img_path, points, body_ids,
            title=f"WX250s Point Cloud  |  {len(points):,} pts  |  {args.density:.0f} pts/cm$^2$"
        )
        print(f"  PNG: {pc_img_path}")

    # Rendered image
    if not args.no_render:
        print("\nRendering views...")
        images = render_views(model, data)
        render_path = os.path.join(args.output, f"{args.prefix}_render.png")
        save_render(
            render_path, images,
            title=f"WX250s  |  gripper={gripper:.0%} open"
        )
        print(f"  PNG: {render_path}")

    # Interactive viewer
    if args.show:
        launch_viewer(model, data)

    print("\nDone!")


if __name__ == "__main__":
    main()
