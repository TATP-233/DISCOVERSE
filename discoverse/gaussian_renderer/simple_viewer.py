import glfw
import OpenGL.GL as gl
import numpy as np
import torch
import argparse
import os
import sys
import time
from scipy.spatial.transform import Rotation

# Add parent dir to path to allow importing gaussian_renderer as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gaussian_renderer.util_gau import load_ply
from gaussian_renderer.batch_rasterization import batch_render
from gaussian_renderer.gaussiandata import GaussianData

class SimpleViewer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.window = None
        
        self.gaussians = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Camera
        self.cam_lookat = np.array([0., 0., 0.])
        self.cam_dist = 3.0
        self.cam_azimuth = 0.0
        self.cam_elevation = 0.0
        self.fov = 45.0
        
        # Mouse
        self.mouse_pressed = {
            glfw.MOUSE_BUTTON_LEFT: False,
            glfw.MOUSE_BUTTON_RIGHT: False,
            glfw.MOUSE_BUTTON_MIDDLE: False
        }
        self.last_mouse_pos = (0, 0)
        
        # Settings
        self.sh_degree = 3
        self.max_sh_degree = 3
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        
        self.init_glfw()
        
    def init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        self.window = glfw.create_window(self.width, self.height, "Simple 3DGS Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
            
        glfw.make_context_current(self.window)
        glfw.set_window_size_callback(self.window, self.resize_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_drop_callback(self.window, self.drop_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        
        # Disable vsync
        glfw.swap_interval(0)

    def load_model(self, path):
        print(f"Loading model from {path}...")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

        try:
            gs_data = load_ply(path)
            self.gaussians = gs_data.to_cuda()
            
            # Ensure SH is 3D (N, K, 3)
            if self.gaussians.sh.dim() == 2:
                 self.gaussians.sh = self.gaussians.sh.reshape(self.gaussians.sh.shape[0], -1, 3)
            
            dim_sh = self.gaussians.sh.shape[1]
            self.max_sh_degree = int(np.sqrt(dim_sh)) - 1
            self.sh_degree = self.max_sh_degree
            
            print(f"Model loaded. Points: {self.gaussians.xyz.shape[0]}, Max SH degree: {self.max_sh_degree}")
            
            # Center camera on model (optional, maybe confusing if model is far from origin)
            # center = torch.mean(self.gaussians.xyz, dim=0).detach().cpu().numpy()
            # self.cam_lookat = center
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    def resize_callback(self, window, width, height):
        self.width = width
        self.height = height
        gl.glViewport(0, 0, width, height)

    def drop_callback(self, window, paths):
        if len(paths) > 0:
            self.load_model(paths[0])

    def update_title(self):
        glfw.set_window_title(self.window, f"Simple 3DGS Viewer - FPS: {self.fps:.1f} - SH: {self.sh_degree}")

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_UP:
                self.sh_degree = min(self.sh_degree + 1, self.max_sh_degree)
                print(f"SH Degree: {self.sh_degree}")
                self.update_title()
            elif key == glfw.KEY_DOWN:
                self.sh_degree = max(self.sh_degree - 1, 0)
                print(f"SH Degree: {self.sh_degree}")
                self.update_title()

    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_pressed[button] = True
        elif action == glfw.RELEASE:
            self.mouse_pressed[button] = False

    def cursor_pos_callback(self, window, xpos, ypos):
        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        self.last_mouse_pos = (xpos, ypos)
        
        if self.mouse_pressed[glfw.MOUSE_BUTTON_LEFT]:
            # Rotate
            self.cam_azimuth -= dx * 0.2
            self.cam_elevation -= dy * 0.2
            self.cam_elevation = np.clip(self.cam_elevation, -89, 89)
            
        elif self.mouse_pressed[glfw.MOUSE_BUTTON_RIGHT] or self.mouse_pressed[glfw.MOUSE_BUTTON_MIDDLE]:
            # Pan
            sensitivity = self.cam_dist * 0.001
            
            camera_rmat = np.array([
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0,  1,  0],
            ])
            R = camera_rmat @ Rotation.from_euler('xyz', [self.cam_elevation * np.pi / 180.0, self.cam_azimuth * np.pi / 180.0, 0.0]).as_matrix()
            
            right = R[:3, 0]
            up = R[:3, 1]
            
            self.cam_lookat -= (right * dx * sensitivity)
            self.cam_lookat += (up * dy * sensitivity)

    def scroll_callback(self, window, xoffset, yoffset):
        self.cam_dist *= (1.0 - yoffset * 0.1)
        self.cam_dist = max(0.01, self.cam_dist)

    def render(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self.gaussians is None:
            return

        # Camera setup
        camera_rmat = np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0],
        ])
        R = camera_rmat @ Rotation.from_euler('xyz', [self.cam_elevation * np.pi / 180.0, self.cam_azimuth * np.pi / 180.0, 0.0]).as_matrix()
        pos = self.cam_lookat + self.cam_dist * R[:3, 2]
        
        cam_pos = pos.reshape(1, 3)
        cam_xmat = R.flatten().reshape(1, 9)
        fovy = np.array([self.fov])
        
        # Slice SH
        current_sh_dim = (self.sh_degree + 1) ** 2
        sh_sliced = self.gaussians.sh[:, :current_sh_dim, :]
        
        render_gaussians = GaussianData(
            self.gaussians.xyz,
            self.gaussians.rot,
            self.gaussians.scale,
            self.gaussians.opacity,
            sh_sliced
        )
        # render_gaussians.device = self.gaussians.device
        
        try:
            color, depth = batch_render(
                render_gaussians,
                cam_pos,
                cam_xmat,
                self.height,
                self.width,
                fovy
            )
            
            img = color[0].detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            # Flip for OpenGL
            img = np.flipud(img)
            
            gl.glDrawPixels(self.width, self.height, gl.GL_RGB, gl.GL_FLOAT, img)
            
        except Exception as e:
            print(f"Render error: {e}")
            time.sleep(0.1)

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            self.render()
            
            self.frame_count += 1
            curr_time = time.time()
            if curr_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (curr_time - self.last_time)
                self.frame_count = 0
                self.last_time = curr_time
                self.update_title()
            
            glfw.swap_buffers(self.window)
            
        glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple 3DGS Viewer")
    parser.add_argument("path", nargs="?", help="Path to .ply file")
    args = parser.parse_args()
    
    viewer = SimpleViewer()
    if args.path:
        viewer.load_model(args.path)
    
    print("Controls:")
    print("  Left Mouse: Rotate")
    print("  Right/Middle Mouse: Pan")
    print("  Scroll: Zoom")
    print("  Up/Down Key: Adjust SH Degree")
    print("  Drag & Drop: Load PLY file")
    
    viewer.run()
