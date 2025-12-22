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
from .config import *

class AvFallbackEncoder:
    def __init__(self, width, height, fps=FPS, bitrate=BITRATE, gop=GOP):
        self.width = width
        self.height = height
        self.output = av.open('pipe:', 'w', format='h264')
        self.stream = self.output.add_stream(CODEC, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = PIX_FMT
        self.stream.bit_rate = int(bitrate.replace('M', '000000'))
        
        self.stream.options = {
            'preset': PRESET, 'tune': 'ull', 'zerolatency': '1',
            'delay': '0', 'bframes': '0', 'profile': 'high'
        }
        
        if gop <= 1:
            self.stream.options['forced-idr'] = '1'
        else:
            self.stream.options['g'] = str(gop)
            self.stream.options['keyint_min'] = str(gop)
        
    def encode_frame(self, tensor_cuda: torch.Tensor):
        # 强制 CPU 拷贝
        frame_cpu_np = tensor_cuda.cpu().numpy()
        frame = av.VideoFrame.from_ndarray(frame_cpu_np, format='rgb24')
        return [bytes(p) for p in self.stream.encode(frame)]

    def flush(self):
        return [bytes(p) for p in self.stream.encode()]
