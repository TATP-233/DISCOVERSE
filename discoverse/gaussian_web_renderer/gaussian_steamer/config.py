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

# config.py

# 渲染分辨率 (H, W)
# 注意：如果使用 video/test.mp4 作为源，建议根据视频实际分辨率调整，或者在 renderer 中进行 resize
RENDER_RESOLUTION = (480, 640) 
# RENDER_RESOLUTION = (1080, 1920) 
# RENDER_RESOLUTION = (2160, 3840) 

FPS = 30 # 假设测试视频是 30fps，或者根据需要调整
GOP = 8 # 默认 GOP 长度，1 为全 I 帧
CODEC = 'h264_nvenc'
PIX_FMT = 'yuv420p' # NVENC 要求
BITRATE = '5M' # 5 Mbps
PRESET = 'llhp' # low latency high performance
