import cv2
import time
import numpy as np

import torch
import torchvision.transforms as transforms

from model_building import SynergyNet

from FaceBoxes import FaceBoxes
from utils.render import render
from utils.ddfa import ToTensor, Normalize
from utils.inference import (
    crop_img,
    predict_sparseVert,
    draw_landmarks,
    predict_denseVert,
    predict_pose,
    draw_axis,
)

import os

os.chdir("/home/tatp/ws/graphics/SynergyNet")


class HeadPose:
    IMG_SIZE = 120

    def __init__(self) -> None:
        class Args:
            arch = "mobilenet_v2"
            devices_id = [0]
            img_size = self.IMG_SIZE

        checkpoint_fp = "pretrained/best.pth.tar"
        checkpoint = torch.load(
            checkpoint_fp, map_location=lambda storage, loc: storage
        )["state_dict"]

        self.model = SynergyNet(Args)
        model_dict = self.model.state_dict()

        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace("module.", "")] = checkpoint[k]

        self.model.load_state_dict(model_dict, strict=False)
        self.model = self.model.cuda()
        self.model.eval()

        # face detector
        self.face_boxes = FaceBoxes()

        # preparation
        self.transform = transforms.Compose(
            [ToTensor(), Normalize(mean=127.5, std=128)]
        )

    def process_frame(self, img_ori, if_draw_axis=True):
        # crop faces
        rects = self.face_boxes(img_ori)
        poses = []
        for _, rect in enumerate(rects):
            roi_box = rect

            # enlarge the bbox a little and do a square crop
            HCenter = (rect[1] + rect[3]) / 2
            WCenter = (rect[0] + rect[2]) / 2
            side_len = roi_box[3] - roi_box[1]
            margin = side_len * 1.2 // 2
            roi_box[0], roi_box[1], roi_box[2], roi_box[3] = (
                WCenter - margin,
                HCenter - margin,
                WCenter + margin,
                HCenter + margin,
            )

            img = crop_img(img_ori, roi_box)
            img = cv2.resize(
                img,
                dsize=(self.IMG_SIZE, self.IMG_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

            input = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                input = input.cuda()
                param = self.model.forward_test(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # inferences
            lmks = predict_sparseVert(param, roi_box, transform=True)
            angles, translation = predict_pose(param, roi_box)

            poses.append([angles, translation, lmks])

        img_ori_copy = img_ori.copy()

        # face orientation
        img_axis_plot = img_ori_copy
        if if_draw_axis and len(poses):
            for angles, translation, lmks in poses:
                img_axis_plot = draw_axis(
                    img_axis_plot,
                    angles[0],
                    angles[1],
                    angles[2],
                    translation[0],
                    translation[1],
                    size=50,
                    pts68=lmks,
                )

        if len(poses) != 0:
            return img_axis_plot, poses
        else:
            return img_axis_plot, None


if __name__ == "__main__":

    hp = HeadPose()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame, pose = hp.process_frame(cv2.flip(frame, 1))
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
            break

    cap.release()
    cv2.destroyAllWindows()
