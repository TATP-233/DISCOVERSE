import os
import time
import json
import zmq
import torch
import numpy as np
import struct
from discoverse.gaussian_renderer.gs_renderer import GSRenderer
from discoverse.gaussian_web_renderer.gaussian_steamer.encoder import vEncoder

class GaussianRenderingServer:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")
        
        self.renderer = None
        self.encoder = None
        self.device = "cuda"
        
        print(f"Gaussian Rendering Server listening on port {port}...")

    def run(self):
        while True:
            try:
                message = self.socket.recv()
            except Exception as e:
                print(f"Socket receive error: {e}")
                break

            try:
                if message.startswith(b'{'):
                    self.handle_init(message)
                else:
                    self.handle_frame(message)
            except Exception as e:
                print(f"Error processing message: {e}")

    def handle_init(self, message):
        print("Received Init request")
        data = json.loads(message.decode('utf-8'))
        
        models_dict = data['models_dict']
        active_bodies = data['active_bodies']
        
        self.renderer = GSRenderer(models_dict)
        
        objects_info = []
        for name in active_bodies:
            if name in self.renderer.gaussian_start_indices:
                start = self.renderer.gaussian_start_indices[name]
                end = self.renderer.gaussian_end_indices[name]
                objects_info.append((name, start, end))
        
        self.renderer.set_objects_mapping(objects_info)
        
        print("Renderer initialized.")
        self.socket.send(b'OK')

    def handle_frame(self, message):
        if self.renderer is None:
            self.socket.send(b'Error: Not Initialized')
            return

        offset = 0
        header_fmt = 'iiii'
        header_size = struct.calcsize(header_fmt)
        num_bodies, num_cams, width, height = struct.unpack_from(header_fmt, message, offset)
        offset += header_size
        
        body_data_size = num_bodies * 7 * 4
        body_data = np.frombuffer(message, dtype=np.float32, count=num_bodies*7, offset=offset)
        body_data = body_data.reshape(num_bodies, 7)
        offset += body_data_size
        
        pos = body_data[:, :3]
        quat = body_data[:, 3:]
        
        cam_data_size = num_cams * 13 * 4
        cam_data = np.frombuffer(message, dtype=np.float32, count=num_cams*13, offset=offset)
        cam_data = cam_data.reshape(num_cams, 13)
        offset += cam_data_size
        
        cam_pos = cam_data[:, :3]
        cam_xmat = cam_data[:, 3:12]
        fovy_arr = cam_data[:, 12]
        
        self.renderer.update_gaussian_properties(pos, quat)
        
        render_width = width
        render_height = height
        
        rgb_tensor, depth_tensor = self.renderer.render_batch(
            cam_pos, cam_xmat, render_height, render_width, fovy_arr
        )
        
        if num_cams > 1:
            final_image_tensor = torch.cat([rgb_tensor[i] for i in range(num_cams)], dim=1)
            enc_width = width * num_cams
            enc_height = height
        else:
            final_image_tensor = rgb_tensor[0]
            enc_width = width
            enc_height = height
            
        final_image_tensor = (final_image_tensor * 255).clamp(0, 255).to(torch.uint8)
        
        if self.encoder is None or self.encoder.width != enc_width or self.encoder.height != enc_height:
            print(f"Initializing Encoder: {enc_width}x{enc_height}")
            self.encoder = vEncoder(enc_width, enc_height, fps=30)
            
        encoded_packets = self.encoder.encode_frame(final_image_tensor)
        encoded_bytes = b''.join(encoded_packets)
        self.socket.send(encoded_bytes)

if __name__ == "__main__":
    server = GaussianRenderingServer()
    server.run()
