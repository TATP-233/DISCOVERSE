# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2025 Yufei Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from .config import *

try:
    import PyNvCodec as nvc
    import PytorchNvCodec as pnvc
    HAS_VPF = True
except (ImportError, RuntimeError):
    HAS_VPF = False

class VpfGpuEncoder:
    """
    基于 NVIDIA VideoProcessingFramework 的真正零拷贝编码器。
    数据流: GPU Tensor -> VPF Surface (GPU) -> NVENC (GPU) -> Bitstream (CPU)
    全程无像素数据的 CPU 拷贝。
    """
    def __init__(self, width, height, fps=FPS, bitrate=BITRATE, gop=GOP):
        if not HAS_VPF:
            raise ImportError("NVIDIA VideoProcessingFramework (VPF) is not installed.")

        self.width = width
        self.height = height
        self.gpu_id = 0 # 默认使用 GPU 0
        
        print(f"[{self.__class__.__name__}] Initializing VPF Zero-Copy Encoder...")

        # 1. 配置 NVENC 参数
        # 码率转换: '5M' -> 5000000
        bitrate_val = int(bitrate.replace('M', '000000'))
        
        opts = {
            'preset': 'll',          # Low Latency
            'codec': 'h264',         # H.264
            's': f'{width}x{height}',
            'bitrate': str(bitrate_val),
            'fps': str(fps),
            'bf': '0',               # No B-frames (关键：低延迟)
            'gop': str(gop),         # GOP size
            'profile': 'high',
        }
        
        try:
            # 指定编码器输入格式为 NV12 (NVENC 原生支持)
            self.nv_enc = nvc.PyNvEncoder(opts, self.gpu_id, nvc.PixelFormat.NV12)
            print(f"[{self.__class__.__name__}] ✅ PyNvEncoder initialized.")
        except Exception as e:
            print(f"[{self.__class__.__name__}] ❌ Failed to init PyNvEncoder: {e}")
            raise e

        # 2. 配置颜色空间转换器 (RGB -> YUV420 -> NV12)
        # 直接 RGB -> NV12 不支持，需两步转换
        self.nv_cvt_1 = nvc.PySurfaceConverter(
            width, height,
            nvc.PixelFormat.RGB,
            nvc.PixelFormat.YUV420,
            self.gpu_id
        )
        self.nv_cvt_2 = nvc.PySurfaceConverter(
            width, height,
            nvc.PixelFormat.YUV420,
            nvc.PixelFormat.NV12,
            self.gpu_id
        )
        self.cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        print(f"[{self.__class__.__name__}] ✅ PySurfaceConverter (RGB->YUV420->NV12) initialized.")

        # 3. 预分配输入 Surface (用于接收 Tensor 数据)
        self.src_surface = nvc.Surface.Make(nvc.PixelFormat.RGB, width, height, self.gpu_id)

        # 4. 预分配输出 Packet 缓冲区 (用于接收压缩后的比特流)
        self.packet = np.ndarray(shape=(0), dtype=np.uint8)

    def encode_frame(self, tensor_cuda: torch.Tensor):
        """
        tensor_cuda: (H, W, 3), uint8, cuda
        """
        if not tensor_cuda.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        
        # 确保内存连续，否则 VPF 无法读取
        if not tensor_cuda.is_contiguous():
            tensor_cuda = tensor_cuda.contiguous()

        # Step 1: 将 PyTorch Tensor 数据拷贝到 VPF Surface
        # 使用 PytorchNvCodec.TensorToDptr 直接在 GPU 上拷贝
        # RGB 格式: width * 3 bytes per row
        pnvc.TensorToDptr(
            tensor_cuda, 
            self.src_surface.PlanePtr(0).GpuMem(), 
            self.width * 3, 
            self.height, 
            self.src_surface.Pitch(0), 
            1
        )
        
        # Step 2: 颜色空间转换 RGB -> YUV420 -> NV12 (在 GPU 上执行)
        yuv420_surface = self.nv_cvt_1.Execute(self.src_surface, self.cc_ctx)
        dst_surface = self.nv_cvt_2.Execute(yuv420_surface, self.cc_ctx)
        
        # Step 3: 编码 (在 GPU 上执行)
        # 结果写入 self.packet (这是压缩后的数据，必须回传 CPU 发送)
        success = self.nv_enc.EncodeSingleSurface(dst_surface, self.packet)
        
        if success:
            return [bytes(self.packet)]
        return []

    def flush(self):
        # VPF flush 逻辑
        packets = []
        while True:
            success = self.nv_enc.Flush(self.packet)
            if success:
                packets.append(bytes(self.packet))
            else:
                break
        return packets
