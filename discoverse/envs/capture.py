import os
import cv2
import mediapy
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs.simulator import BaseConfig
from discoverse.envs import SimulatorBase

from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR


class CaptureCfg(BaseConfig):
    robot = "capture"
    mjcf_file_path = "mjcf/capture.xml"
    decimation = 1
    timestep = 0.02
    sync = True
    headless = False
    render_set = {"fps": 50, "width": 1920, "height": 1080}
    obs_rgb_cam_id = -1


class CaptureBase(SimulatorBase):
    def __init__(self, config: CaptureCfg):
        super().__init__(config)
        self.save_dir = os.path.join(
            DISCOVERSE_ROOT_DIR, f"data/capture/{config.expreriment}"
        )
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        img_save_dir = os.path.join(self.save_dir, "images")
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)
        self.cap_id = 0

        cameras_extrinsic_file = os.path.join(self.save_dir, "images.txt")
        self.ext_fp = open(cameras_extrinsic_file, "w")
        self.ext_cam_fp = open(
            cameras_extrinsic_file.replace("images.txt", "cam_pose.txt"), "w"
        )
        self.obj_posi_fp = open(
            cameras_extrinsic_file.replace("images.txt", "obj_pose.txt"), "w"
        )

        cameras_intrinsic_file = os.path.join(self.save_dir, "cameras.txt")
        with open(cameras_intrinsic_file, "w") as fp:
            fp.write(
                "1 PINHOLE {} {} {} {} {} {}\n".format(
                    self.config.render_set["width"],
                    self.config.render_set["height"],
                    self.config.render_set["height"]
                    / 2.0
                    / np.tan(45 / 2 * np.pi / 180),
                    self.config.render_set["height"]
                    / 2.0
                    / np.tan(45 / 2 * np.pi / 180),
                    self.config.render_set["width"] / 2,
                    self.config.render_set["height"] / 2,
                )
            )

    def capImg(self):
        self.cap_id += 1
        img_name = "img_{:03d}.png".format(self.cap_id)
        img_file = os.path.join(self.save_dir, "images/{}".format(img_name))
        posi, quat_wxyz_ = self.getCameraPose(self.cam_id)
        quat_xyzw_ = quat_wxyz_[[1, 2, 3, 0]]
        rmat = (
            Rotation.from_quat(quat_xyzw_).as_matrix()
            @ Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        )
        quat_xyzw = Rotation.from_matrix(rmat).as_quat()

        qvec = Rotation.from_matrix(rmat.T).as_quat()
        tvec = -rmat.T @ posi

        self.ext_fp.write(
            "{} {} {} {} {} {} {} {} {} {}\n".format(
                self.cap_id,
                qvec[3],
                qvec[0],
                qvec[1],
                qvec[2],
                tvec[0],
                tvec[1],
                tvec[2],
                self.cam_id + 2,
                img_name,
            )
        )

        self.ext_cam_fp.write(
            '<site pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                posi[0],
                posi[1],
                posi[2],
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )
        )
        self.obj_posi_fp.write(
            "{} {} {} {}\n".format(
                img_name,
                self.mj_model.body(self.config.expreriment).pos[0],
                self.mj_model.body(self.config.expreriment).pos[1],
                self.mj_model.body(self.config.expreriment).pos[2],
            )
        )
        cv2.imwrite(img_file, cv2.cvtColor(self.img_rgb_obs, cv2.COLOR_RGB2BGR))

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {}
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None


if __name__ == "__main__":
    # cap_type = "auto_cap"
    # cap_type = "move_obj"
    cap_type = "cap_movie"

    cfg = CaptureCfg()
    cfg.mjcf_file_path = "mjcf/capture.xml"
    cfg.sync = False
    cfg.obs_rgb_cam_id = -1

    cap = CaptureBase(cfg)

    if cap_type == "auto_cap":
        cap.free_camera.distance = 0.5
        # cap.free_camera.lookat = np.array([0.15, 0, 0])
        cap.free_camera.azimuth = 0
        cap.free_camera.elevation = -85
        while cap.running:
            cap.free_camera.azimuth += 10.0
            if cap.free_camera.azimuth > 360:
                cap.free_camera.azimuth -= 360
                cap.free_camera.elevation += 15
                if cap.free_camera.elevation > 89:
                    break
            cap.step()
            cap.capImg()
            print("Captured image: img_{:03d}.png".format(cap.cap_id))
    elif cap_type == "move_obj":
        cap.cam_id = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    cap.mj_model.body(cfg.expreriment).pos[0] = i * 0.1 - 0.2
                    cap.mj_model.body(cfg.expreriment).pos[1] = j * 0.1 - 0.2
                    cap.mj_model.body(cfg.expreriment).pos[2] = -k * 0.1 - 0.5
                    cap.step()
                    cap.capImg()
                    print("Captured image: img_{:03d}.png".format(cap.cap_id))
        cap.ext_fp.close()
        cap.ext_cam_fp.close()
        cap.obj_posi_fp.close()

    elif cap_type == "cap_movie":
        movie_time_s = 3.0
        cap.free_camera.distance = 5.0
        cap.free_camera.lookat = np.array([0, -1.0, 0.5])
        cap.free_camera.azimuth = -180
        cap.free_camera.elevation = -35

        img_lst = []
        while cap.running:
            cap.free_camera.azimuth += 360 / cfg.render_set["fps"] / movie_time_s
            cap.step()
            img_lst.append(cap.img_rgb_obs.copy())
            if len(img_lst) > movie_time_s * cfg.render_set["fps"]:
                break

        save_path = "/home/tatp/ws/GreatWall/dlab-sim/data/iros_talk"
        mediapy.write_video(
            os.path.join(save_path, f"{cfg.expreriment}.mp4"),
            img_lst,
            fps=cfg.render_set["fps"],
        )
