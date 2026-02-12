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

from .encoder_gpu import VpfGpuEncoder, HAS_VPF
from .encoder_cpu import AvFallbackEncoder

# ============================================================================
# 导出类
# ============================================================================
if HAS_VPF:
    vEncoder = VpfGpuEncoder
else:
    class vEncoder(AvFallbackEncoder):
        def __init__(self, *args, **kwargs):
            # ANSI colors
            RED = '\033[91m'
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            print(f"\n{RED}[Error] NVIDIA VideoProcessingFramework (VPF) NOT FOUND!{RESET}")
            print(f"{YELLOW}You requested TRUE ZERO-COPY encoding, but the required library is missing.")
            print(f"Please install VPF from source: https://github.com/NVIDIA/VideoProcessingFramework{RESET}")
            print(f"{YELLOW}Running in COMPATIBILITY MODE (CPU Copy) to prevent crash...{RESET}\n")
            super().__init__(*args, **kwargs)
