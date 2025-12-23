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

import av
import torch
import numpy as np
from fractions import Fraction
from .config import *

class AvFallbackEncoder:
    def __init__(self, width, height, fps=FPS, bitrate=BITRATE, gop=GOP):
        self.width = width
        self.height = height
        
        # Use CodecContext directly for raw H.264 encoding to match GPU encoder behavior
        codec = av.Codec('libx264', 'w')
        self.ctx = av.CodecContext.create(codec)
        
        self.ctx.width = width
        self.ctx.height = height
        self.ctx.pix_fmt = 'yuv420p'
        self.ctx.time_base = Fraction(1, int(fps))
        self.ctx.framerate = Fraction(int(fps), 1)
        
        # Bitrate conversion
        if isinstance(bitrate, str):
            bitrate_val = int(bitrate.replace('M', '000000'))
        else:
            bitrate_val = int(bitrate)
        self.ctx.bit_rate = bitrate_val
        
        # x264 options for ultra-low latency
        self.ctx.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'bframes': '0',
            'threads': '1',
        }
        
        if gop <= 1:
            self.ctx.options['keyint'] = '1'
        else:
            self.ctx.options['g'] = str(gop)
            
        self.ctx.open()
        
    def encode_frame(self, tensor_cuda: torch.Tensor):
        # Move to CPU and convert to numpy
        if tensor_cuda.is_cuda:
            frame_cpu_np = tensor_cuda.cpu().numpy()
        else:
            frame_cpu_np = tensor_cuda.numpy()
            
        # Ensure the array has the correct shape (H, W, 3)
        frame = av.VideoFrame.from_ndarray(frame_cpu_np, format='rgb24')
        
        packets = self.ctx.encode(frame)
        return [bytes(p) for p in packets]

    def flush(self):
        return [bytes(p) for p in self.ctx.encode()]
