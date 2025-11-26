'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from discoverse.gaussian_renderer import util_gau
import numpy as np
import torch
from dataclasses import dataclass

# Try to import both rendering backends
try:
    from diff_gaussian_rasterization import GaussianRasterizer as GausssianRasterizer_3d
    DIFF_GAUSSIAN_AVAILABLE = True
except ImportError:
    DIFF_GAUSSIAN_AVAILABLE = False
    print("Warning: diff_gaussian_rasterization not available")

try:
    from gsplat.rendering import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available")

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

class CUDARenderer:
    def __init__(self, w, h, backend="gsplat"):
        """
        Initialize CUDA Renderer with specified backend
        
        Args:
            w: Width of the rendering resolution
            h: Height of the rendering resolution
            backend: Rendering backend to use, either "gsplat" or "diff_gaussian"
        """
        super().__init__()
        
        # Validate backend
        if backend == "gsplat":
            if not GSPLAT_AVAILABLE:
                raise RuntimeError("gsplat backend requested but not available. Please install gsplat.")
            self.backend = "gsplat"
        elif backend == "diff_gaussian":
            if not DIFF_GAUSSIAN_AVAILABLE:
                raise RuntimeError("diff_gaussian backend requested but not available. Please install diff_gaussian_rasterization.")
            self.backend = "diff_gaussian"
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be 'gsplat' or 'diff_gaussian'")
        
        print(f"Using rendering backend: {self.backend}")
        
        raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        self.raster_settings = GaussianRasterizationSettingsStorage(**raster_settings)

        self.depth_render = False
        self.need_rerender = True
        self.render_rgb_img = None
        self.render_depth_img = None

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.need_rerender = True
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            self.gau_env_idx = gau_xyz[0].shape[0]
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gau.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings.sh_degree = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings.scale_modifier = float(modifier)

    def update_gaussian_data_fast(self, start_idx, end_idx, pos_expanded, quat_expanded):
        """
        快速更新高斯数据，直接写入显存
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            pos_expanded: 扩展后的位置数据 (N, 3)
            quat_expanded: 扩展后的旋转数据 (N, 4)
        """
        self.need_rerender = True
        self.gau_xyz_all_cu[start_idx:end_idx] = pos_expanded
        self.gau_rot_all_cu[start_idx:end_idx] = quat_expanded

    def update_gaussian_data_fast_indices(self, indices, pos_expanded, quat_expanded):
        """
        快速更新高斯数据，使用索引直接写入显存
        
        Args:
            indices: 索引张量 (N,)
            pos_expanded: 扩展后的位置数据 (N, 3)
            quat_expanded: 扩展后的旋转数据 (N, 4)
        """
        self.need_rerender = True
        self.gau_xyz_all_cu[indices] = pos_expanded
        self.gau_rot_all_cu[indices] = quat_expanded

    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings.image_height = int(h)
        self.raster_settings.image_width = int(w)

    def update_camera_pose(self, camera: util_gau.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix(self.backend)
        
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()

    def update_camera_pose_from_topic(self, camera: util_gau.Camera, rmat, trans):
        self.need_rerender = True

        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)

        Tmat = np.eye(4)
        Tmat[:3,:3] = rmat
        Tmat[:3,3] = trans
        Tmat[0:3, [1,2]] *= -1
        
        view_matrix = np.linalg.inv(Tmat)

        if self.backend == "diff_gaussian":
            view_matrix[[0,1], :] *= -1

        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings.viewmatrix = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings.campos = torch.tensor(camera.position).float().cuda()
        self.raster_settings.projmatrix = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util_gau.Camera):
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings.tanfovx = hfovx
        self.raster_settings.tanfovy = hfovy

    def draw(self, render_depth=False):
        if not self.need_rerender:
            if render_depth:
                return self.render_depth_img
            else:
                return self.render_rgb_img

        self.need_rerender = False

        if self.backend == "gsplat":
            # Use gsplat backend
            with torch.no_grad():
               
                # Prepare camera matrices
                width = self.raster_settings.image_width
                height = self.raster_settings.image_height
                focal_x = width / (2.0 * self.raster_settings.tanfovx)
                focal_y = height / (2.0 * self.raster_settings.tanfovy)
                
                # For gsplat, we use batch_dims to organize the data
                # - Gaussians: [G, 3], [G, 4], [G, 3], [G], [G, K, 3]
                # - Cameras: [C, 4, 4], [C, 3, 3]
                # Output will be [C, H, W, X]
                
                # Create K matrix - shape [C, 3, 3] with C cameras
                K = torch.tensor([
                    [focal_x, 0.0, width / 2.0],
                    [0.0, focal_y, height / 2.0],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32, device='cuda').unsqueeze(0) # [C, 3, 3]
                
                # Get viewmat (transpose back since we stored transposed) - shape [C, 4, 4]
                viewmat = self.raster_settings.viewmatrix.T.unsqueeze(0)  # [C, 4, 4]
                                
                # Call gsplat rasterization
                # means=[G,3], quats=[G,4], scales=[G,3], opacities=[G], colors=[G,K,3], viewmats=[C,4,4], Ks=[C,3,3]
                # Output: renders=[C,H,W,X], alphas=[C,H,W,1]
                renders, alphas, meta = rasterization(
                    means=self.gaussians.xyz,  # [G, 3]
                    quats=self.gaussians.rot,  # [G, 4]
                    scales=self.gaussians.scale,  # [G, 3]
                    opacities=self.gaussians.opacity,  # [G]
                    colors=self.gaussians.sh,  # [G, K, 3]
                    viewmats=viewmat,  # [C, 4, 4]
                    Ks=K,  # [C, 3, 3]
                    width=width,
                    height=height,
                    sh_degree=self.raster_settings.sh_degree,
                    render_mode="RGB+D" if render_depth else "RGB",
                    packed=False,
                )
                # renders shape: [C, height, width, X]
                # Remove batch and camera dimensions since we usually have 1 batch and 1 camera
                renders = renders.squeeze(0).squeeze(0)  # [height, width, X]
                
                if render_depth:
                    # RGB+D mode: last channel is depth
                    color_img = renders[..., :3]  # [height, width, 3]
                    depth_img = renders[..., 3:4]  # [height, width, 1]
                    self.render_depth_img = depth_img.contiguous().cpu().numpy()
                    self.render_rgb_img = (255. * torch.clamp(color_img, 0.0, 1.0)).to(torch.uint8).contiguous().cpu().numpy()
                else:
                    # RGB mode only
                    color_img = renders  # [height, width, 3]
                    self.render_rgb_img = (255. * torch.clamp(color_img, 0.0, 1.0)).to(torch.uint8).contiguous().cpu().numpy()
                    
        else:  # diff_gaussian backend
            rasterizer = GausssianRasterizer_3d(raster_settings=self.raster_settings)

            with torch.no_grad():
                color_img, radii, depth_img, _alpha = rasterizer(
                    means3D = self.gaussians.xyz,
                    means2D = None,
                    shs = self.gaussians.sh,
                    colors_precomp = None,
                    opacities = self.gaussians.opacity,
                    scales = self.gaussians.scale,
                    rotations = self.gaussians.rot,
                    cov3D_precomp = None
                )

                self.render_depth_img = depth_img.permute(1, 2, 0).contiguous().cpu().numpy()
                self.render_rgb_img = (255. * torch.clamp(color_img, 0.0, 1.0)).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

        if render_depth:
            return self.render_depth_img
        else:
            return self.render_rgb_img
