import torch
import torch.nn.functional as F
import cv2
import numpy as np

from .raft.raft import RAFT
import torch
import argparse

# 光流计算
class FlowCompute:
    def __init__(
        self,
        method="rgb",
        model="default",
        small=True,
        mixed_precision=False,
        alternate_corr=False,
        device="cuda",
    ):
        self.method = method
        self.device = device

        if self.method == "raft":
            args = argparse.Namespace(
                model=model,
                small=small,
                mixed_precision=mixed_precision,
                alternate_corr=alternate_corr,
            )
            self.model = torch.nn.DataParallel(RAFT(args))
            self.model.load_state_dict(torch.load(args.model))

            self.model = self.model.module
            self.model.to(self.device)
            self.model.eval()

    def compute(self, prev_img, next_img):
        if self.method == "raft":
            with torch.no_grad():
                image1 = (
                    torch.from_numpy(prev_img)
                    .permute(2, 0, 1)
                    .float()[None]
                    .to(self.device)
                )
                image2 = (
                    torch.from_numpy(next_img)
                    .permute(2, 0, 1)
                    .float()[None]
                    .to(self.device)
                )
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        elif self.method == "rgb":
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                next_gray, prev_gray, None, 0.4, 5, 30, 5, 7, 1.5, 0
            )
        return flow

    # 光流扭曲
    def warp_forward(self, x, flo):
        x = torch.from_numpy(x).permute(2, 0, 1).to(self.device)
        x = x[None, ...].float()
        x /= 255
        flo = torch.from_numpy(flo).permute(2, 0, 1).to(self.device)
        flo = flo[None, ...].float()

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        mask = torch.ones(x.size())

        if self.device == "cuda":
            grid = grid.cuda()
            mask = mask.to(self.device)
        # Invert the flow by multiplying by -1
        vgrid = grid - flo  # Change here
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=False)
        output = output[0].permute(1, 2, 0)
        output = output.cpu().numpy()
        output = (output * 255).astype(np.uint8)
        return output
