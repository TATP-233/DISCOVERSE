import zmq
import json
import time
import numpy as np
import struct
import torch
from discoverse.gaussian_renderer.gs_renderer_mujoco import GSRendererMuJoCo
from discoverse.gaussian_web_renderer.gaussian_steamer.decoder import H264Decoder

class GSRendererRemote(GSRendererMuJoCo):
    def __init__(self, models_dict: dict, server_ip="127.0.0.1", server_port=5555, monitor_latency=False):
        super().__init__(models_dict)
        self.models_dict = models_dict
        self.server_ip = server_ip
        self.server_port = server_port
        self.monitor_latency = monitor_latency
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        
        self.decoder = H264Decoder()
        self.last_pos = None
        self.last_quat = None
        self.is_initialized_on_server = False

    def init_renderer(self, mj_model):
        super().init_renderer(mj_model)
        
        mapping_list = []
        for i in range(len(self.gs_body_ids)):
            start = int(self.gs_idx_start[i])
            end = int(self.gs_idx_end[i])
            mapping_list.append(("body_" + str(i), start, end))
            
        init_data = {
            "models_dict": self.models_dict,
            "objects_info": mapping_list
        }
        
        print("Sending Init to Server...")
        self.socket.send(json.dumps(init_data).encode('utf-8'))
        resp = self.socket.recv()
        if resp == b'OK':
            print("Server Initialized.")
            self.is_initialized_on_server = True
        else:
            print(f"Server Init Failed: {resp}")

    def update_gaussians(self, mj_data):
        if not hasattr(self, 'gs_idx_start') or len(self.gs_idx_start) == 0:
            return
        self.last_pos = mj_data.xpos[self.gs_body_ids]
        self.last_quat = mj_data.xquat[self.gs_body_ids]

    def render(self, mj_model, mj_data, cam_ids, width, height, free_camera=None):
        if not self.is_initialized_on_server or self.last_pos is None:
            return {}

        num_bodies = len(self.last_pos)
        poses = np.hstack([self.last_pos, self.last_quat]).astype(np.float32)
        
        cam_params_list = []
        fixed_cam_ids = [cid for cid in cam_ids if cid != -1]
        
        if len(fixed_cam_ids) > 0:
            fixed_cam_indices = np.array(fixed_cam_ids)
            cam_pos_fixed = mj_data.cam_xpos[fixed_cam_indices]
            cam_xmat_fixed = mj_data.cam_xmat[fixed_cam_indices]
            fovy_fixed = mj_model.cam_fovy[fixed_cam_indices]
            
            for i in range(len(fixed_cam_ids)):
                cam_params_list.append({
                    'pos': cam_pos_fixed[i],
                    'xmat': cam_xmat_fixed[i],
                    'fovy': fovy_fixed[i]
                })
        
        if -1 in cam_ids:
            if free_camera is None:
                raise ValueError("free_camera must be provided")
            from scipy.spatial.transform import Rotation
            camera_rmat = np.array([[0,0,-1],[-1,0,0],[0,1,0]])
            rotation_matrix = camera_rmat @ Rotation.from_euler('xyz', [free_camera.elevation * np.pi / 180.0, free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = free_camera.lookat + free_camera.distance * rotation_matrix[:3,2]
            cam_params_list.append({
                'pos': camera_position,
                'xmat': rotation_matrix.flatten(),
                'fovy': mj_model.vis.global_.fovy
            })
            
        num_cams = len(cam_params_list)
        if num_cams == 0:
            return {}

        cam_data_arr = []
        for cam in cam_params_list:
            cam_data_arr.extend(cam['pos'])
            cam_data_arr.extend(cam['xmat'])
            cam_data_arr.append(cam['fovy'])
        cam_data_np = np.array(cam_data_arr, dtype=np.float32)
        
        header = struct.pack('iiii', num_bodies, num_cams, width, height)
        message = header + poses.tobytes() + cam_data_np.tobytes()
        
        t0 = time.time()
        self.socket.send(message)
        encoded_data = self.socket.recv()
        t1 = time.time()
        
        if self.monitor_latency:
            print(f"\rLatency: {(t1-t0)*1000:.2f} ms", end="")
        
        decoded_frame = None
        if encoded_data:
            decoded_frame = self.decoder.decode(encoded_data)
            if decoded_frame is None:
                # print(f"Warning: Decoder returned None. Encoded data size: {len(encoded_data)} bytes")
                pass
        else:
            print("Warning: Received empty encoded data from server.")

        if decoded_frame is None:
            full_w = num_cams * width
            full_h = height
            decoded_frame_rgb = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        else:
            decoded_frame_rgb = decoded_frame[..., ::-1].copy()
        
        decoded_torch = torch.from_numpy(decoded_frame_rgb)
        if decoded_torch.shape[1] != num_cams * width:
            print(f"Warning: Decoded frame width {decoded_torch.shape[1]} != expected {num_cams * width}")
        
        results = {}
        current_cam_indices = fixed_cam_ids + ([-1] if -1 in cam_ids else [])
        single_w = decoded_torch.shape[1] // num_cams
        
        for i, cid in enumerate(current_cam_indices):
            w_start = i * single_w
            w_end = (i + 1) * single_w
            img_slice = decoded_torch[:, w_start:w_end, :]
            # depth_slice = torch.zeros((height, single_w, 1), dtype=torch.float32)
            results[cid] = (img_slice, None)
            
        return results
