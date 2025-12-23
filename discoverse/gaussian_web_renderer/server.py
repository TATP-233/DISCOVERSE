import os
import time
import json
import asyncio
import zmq
import zmq.asyncio
import torch
import numpy as np
import struct
from concurrent.futures import ThreadPoolExecutor
from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.gaussian_renderer.gs_renderer import GSRenderer
from discoverse.gaussian_web_renderer.gaussian_steamer.encoder import vEncoder

class GaussianRenderingServer:
    def __init__(self, port=5555):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        
        self.clients = {}
        self.max_clients = 8
        self.executor = ThreadPoolExecutor(max_workers=self.max_clients)
        self.device = "cuda"
        
        print(f"Gaussian Rendering Server listening on port {port}...")

    async def run(self):
        asyncio.create_task(self.cleanup_loop())
        while True:
            try:
                parts = await self.socket.recv_multipart()
                if len(parts) < 3:
                    continue
                
                identity = parts[0]
                # parts[1] is empty delimiter
                message = parts[2]

                if message.startswith(b'{'):
                    asyncio.create_task(self.handle_init(identity, message))
                else:
                    asyncio.create_task(self.handle_frame(identity, message))
            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(5)
            current_time = time.time()
            to_remove = []
            for identity, client_data in self.clients.items():
                if current_time - client_data.get('last_activity', current_time) > 30:
                    to_remove.append(identity)
            
            for identity in to_remove:
                print(f"Client {identity} inactive for 30s. Cleaning up.")
                self.clients.pop(identity, None)

    def _resolve_path(self, path):
        if os.path.exists(path):
            return path
        
        # Normalize path separators
        path_norm = path.replace('\\', '/')
        
        # Try to resolve path relative to server's assets directory
        # We look for common keywords in the path
        keywords = ['3dgs', 'models', 'assets']
        
        for kw in keywords:
            token = f'/{kw}/'
            if token in path_norm:
                suffix = path_norm.split(token)[-1]
                
                if kw == '3dgs':
                    # If '3dgs' is found, we assume it's inside the assets dir
                    candidate = os.path.join(DISCOVERSE_ASSETS_DIR, '3dgs', suffix)
                else:
                    # For 'models' or 'assets', we treat the suffix as relative to assets dir
                    candidate = os.path.join(DISCOVERSE_ASSETS_DIR, suffix)
                
                if os.path.exists(candidate):
                    # print(f"Path resolved: {path} -> {candidate}")
                    return candidate
                    
        print(f"Warning: Could not resolve path: {path}")
        return path

    async def handle_init(self, identity, message):
        if len(self.clients) >= self.max_clients and identity not in self.clients:
            print(f"Server busy. Rejecting {identity}")
            await self.socket.send_multipart([identity, b'', b'Busy'])
            return

        print(f"Received Init request from {identity}")
        
        loop = asyncio.get_running_loop()
        
        def init_task():
            data = json.loads(message.decode('utf-8'))
            models_dict = data['models_dict']
            active_bodies = data['active_bodies']
            
            # Resolve paths for server environment
            for name, path in models_dict.items():
                models_dict[name] = self._resolve_path(path)
            
            renderer = GSRenderer(models_dict)
            
            objects_info = []
            for name in active_bodies:
                if name in renderer.gaussian_start_indices:
                    start = renderer.gaussian_start_indices[name]
                    end = renderer.gaussian_end_indices[name]
                    objects_info.append((name, start, end))
            
            renderer.set_objects_mapping(objects_info)
            return renderer

        try:
            renderer = await loop.run_in_executor(self.executor, init_task)
            
            self.clients[identity] = {
                'renderer': renderer,
                'encoder': None,
                'stream': torch.cuda.Stream(),
                'last_activity': time.time()
            }
            
            print(f"Renderer initialized for {identity}.")
            await self.socket.send_multipart([identity, b'', b'OK'])
        except Exception as e:
            print(f"Init failed for {identity}: {e}")
            await self.socket.send_multipart([identity, b'', f'Error: {str(e)}'.encode()])

    async def handle_frame(self, identity, message):
        if identity not in self.clients:
            await self.socket.send_multipart([identity, b'', b'Error: Not Initialized'])
            return

        self.clients[identity]['last_activity'] = time.time()

        loop = asyncio.get_running_loop()
        
        try:
            encoded_bytes = await loop.run_in_executor(
                self.executor, 
                self._render_worker, 
                identity, 
                message
            )
            await self.socket.send_multipart([identity, b'', encoded_bytes])
        except Exception as e:
            print(f"Frame processing failed for {identity}: {e}")
            # traceback.print_exc()
            # Critical: Send empty response to prevent client (REQ socket) from hanging
            try:
                await self.socket.send_multipart([identity, b'', b''])
            except Exception as send_e:
                print(f"Failed to send error response to {identity}: {send_e}")

    def _render_worker(self, identity, message):
        client_data = self.clients[identity]
        renderer = client_data['renderer']
        stream = client_data['stream']
        
        with torch.cuda.stream(stream):
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
            
            renderer.update_gaussian_properties(pos, quat)
            
            render_width = width
            render_height = height
            
            rgb_tensor, depth_tensor = renderer.render_batch(
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
        
        # Synchronize stream to ensure rendering is complete before encoding
        # This prevents artifacts (mosaic) when encoder reads incomplete memory
        stream.synchronize()
            
        encoder = client_data['encoder']
        if encoder is None or encoder.width != enc_width or encoder.height != enc_height:
            print(f"Initializing Encoder for {identity}: {enc_width}x{enc_height}")
            encoder = vEncoder(enc_width, enc_height, fps=30)
            client_data['encoder'] = encoder
            
        encoded_packets = encoder.encode_frame(final_image_tensor)
        encoded_bytes = b''.join(encoded_packets)
        return encoded_bytes

if __name__ == "__main__":
    server = GaussianRenderingServer()
    asyncio.run(server.run())
