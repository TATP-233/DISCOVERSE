import mujoco
import numpy as np

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class MMK2Cfg(BaseConfig):
    mjcf_file_path = "mjcf/mmk2_floor.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080 
    }
    init_state     = {
        "base_position"    : [0.0, 0.0, 0.0],
        "base_orientation" : [1.0, 0.0, 0.0, 0.0],
        "slide_qpos"       : [0.0],
        "head_qpos"        : [0.0, 0.0],
        "lft_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lft_gripper_qpos" : [0.0],
        "rgt_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rgt_gripper_qpos" : [0.0],
    }
    obs_rgb_cam_id   = None
    obs_depth_cam_id = None
    lidar_s2_sim = False
    use_gaussian_renderer = False
    gs_model_dict  = {
        "agv_link"              :   "mobile_chassis/mmk2/agv_link.ply",
        "slide_link"            :   "mobile_chassis/mmk2/slide_link.ply",
        "head_pitch_link"       :   "mobile_chassis/mmk2/head_pitch_link.ply",
        "head_yaw_link"         :   "mobile_chassis/mmk2/head_yaw_link.ply",

        "lft_arm_base"          :   "manipulator/airbot_play/arm_base.ply",
        "lft_arm_link1"         :   "manipulator/airbot_play/link1.ply",
        "lft_arm_link2"         :   "manipulator/airbot_play/link2.ply",
        "lft_arm_link3"         :   "manipulator/airbot_play/link3.ply",
        "lft_arm_link4"         :   "manipulator/airbot_play/link4.ply",
        "lft_arm_link5"         :   "manipulator/airbot_play/link5.ply",
        "lft_arm_link6"         :   "manipulator/airbot_play/link6.ply",
        "lft_finger_left_link"  :   "manipulator/airbot_play/left.ply",
        "lft_finger_right_link" :   "manipulator/airbot_play/right.ply",

        "rgt_arm_base"          :   "manipulator/airbot_play/arm_base.ply",
        "rgt_arm_link1"         :   "manipulator/airbot_play/link1.ply",
        "rgt_arm_link2"         :   "manipulator/airbot_play/link2.ply",
        "rgt_arm_link3"         :   "manipulator/airbot_play/link3.ply",
        "rgt_arm_link4"         :   "manipulator/airbot_play/link4.ply",
        "rgt_arm_link5"         :   "manipulator/airbot_play/link5.ply",
        "rgt_arm_link6"         :   "manipulator/airbot_play/link6.ply",
        "rgt_finger_left_link"  :   "manipulator/airbot_play/left.ply",
        "rgt_finger_right_link" :   "manipulator/airbot_play/right.ply"
    }
    
class MMK2Base(SimulatorBase):
    njq = 28
    njctrl = 19
    wheel_radius = 0.0838
    wheel_distance = 0.189

    def __init__(self, config: MMK2Cfg):
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            if hasattr(self.config, "init_state") and self.config.init_state is not None:
                self.init_joint_pose = np.concatenate([
                    self.config.init_state["base_position"],
                    self.config.init_state["base_orientation"],
                    [0.0, 0.0],
                    self.config.init_state["slide_qpos"],
                    self.config.init_state["head_qpos"],
                    self.config.init_state["lft_arm_qpos"],
                    self.config.init_state["lft_gripper_qpos"],
                    [-self.config.init_state["lft_gripper_qpos"][0]],
                    self.config.init_state["rgt_arm_qpos"],
                    self.config.init_state["rgt_gripper_qpos"],
                    [-self.config.init_state["rgt_gripper_qpos"][0]],
                ])
                self.init_joint_ctrl = np.concatenate([
                    [0.0, 0.0],
                    self.config.init_state["slide_qpos"],
                    self.config.init_state["head_qpos"],
                    self.config.init_state["lft_arm_qpos"],
                    self.config.init_state["lft_gripper_qpos"],
                    self.config.init_state["rgt_arm_qpos"],
                    self.config.init_state["rgt_gripper_qpos"],
                ])
            else:
                raise KeyError("init_state not found in config")
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.njq)
            self.init_joint_pose[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.init_joint_ctrl = np.zeros(self.njctrl)
        
        self.sensor_qpos  = self.mj_data.sensordata[:self.njctrl]
        self.sensor_qvel  = self.mj_data.sensordata[self.njctrl:2*self.njctrl]
        self.sensor_force = self.mj_data.sensordata[2*self.njctrl:3*self.njctrl]

        # base
        self.sensor_base_position    = self.mj_data.sensordata[3*self.njctrl:3*self.njctrl+3]
        self.sensor_base_orientation = self.mj_data.sensordata[3*self.njctrl+3:3*self.njctrl+7]
        self.sensor_base_linear_vel  = self.mj_data.sensordata[3*self.njctrl+7:3*self.njctrl+10]
        self.sensor_base_gyro        = self.mj_data.sensordata[3*self.njctrl+10:3*self.njctrl+13]
        self.sensor_base_acc         = self.mj_data.sensordata[3*self.njctrl+13:3*self.njctrl+16]

        # arm endpoint
        self.sensor_lftarm_ep = self.mj_data.sensordata[3*self.njctrl+16:3*self.njctrl+19]
        self.sensor_lftarm_eo = self.mj_data.sensordata[3*self.njctrl+19:3*self.njctrl+23]
        self.sensor_rgtarm_ep = self.mj_data.sensordata[3*self.njctrl+23:3*self.njctrl+26]
        self.sensor_rgtarm_eo = self.mj_data.sensordata[3*self.njctrl+26:3*self.njctrl+30]

        # wheel
        self.sensor_wheel_qpos = self.sensor_qpos[:2]
        self.sensor_wheel_qvel = self.sensor_qvel[:2]
        self.sensor_wheel_qctrl = self.sensor_force[:2]

        # slide
        self.sensor_slide_qpos = self.sensor_qpos[2:3]
        self.sensor_slide_qvel = self.sensor_qvel[2:3]
        self.sensor_slide_qctrl = self.sensor_force[2:3]

        # head
        self.sensor_head_qpos  = self.sensor_qpos[3:5]
        self.sensor_head_qvel  = self.sensor_qvel[3:5]
        self.sensor_head_qctrl = self.sensor_force[3:5]

        # left arm
        self.sensor_lft_arm_qpos  = self.sensor_qpos[5:11]
        self.sensor_lft_arm_qvel  = self.sensor_qvel[5:11]
        self.sensor_lft_arm_qctrl = self.sensor_force[5:11]

        # left gripper
        self.sensor_lft_gripper_qpos  = self.sensor_qpos[11:12]
        self.sensor_lft_gripper_qvel  = self.sensor_qvel[11:12]
        self.sensor_lft_gripper_ctrl = self.sensor_force[11:12]

        # right arm
        self.sensor_rgt_arm_qpos  = self.sensor_qpos[12:18]
        self.sensor_rgt_arm_qvel  = self.sensor_qvel[12:18]
        self.sensor_rgt_arm_qctrl = self.sensor_force[12:18]

        # right gripper
        self.sensor_rgt_gripper_qpos  = self.sensor_qpos[18:19]
        self.sensor_rgt_gripper_qvel  = self.sensor_qvel[18:19]
        self.sensor_rgt_gripper_ctrl = self.sensor_force[18:19]

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("mj_data.qpos :")
        print("    base      = {}".format(np.array2string(np.hstack([self.sensor_base_position, self.sensor_base_orientation]), separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.sensor_wheel_qpos, separator=', ')))
        print("    slide     = {}".format(np.array2string(self.sensor_slide_qpos, separator=', ')))
        print("    head      = {}".format(np.array2string(self.sensor_head_qpos, separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.sensor_lft_arm_qpos, separator=', ')))
        print("    left  grp = {}".format(np.array2string(self.sensor_lft_gripper_qpos, separator=', ')))
        print("    right arm = {}".format(np.array2string(self.sensor_rgt_arm_qpos, separator=', ')))
        print("    right grp = {}".format(np.array2string(self.sensor_rgt_gripper_qpos, separator=', ')))
        print("    left  end = {}".format(np.array2string(np.hstack([self.sensor_lftarm_ep, self.sensor_lftarm_eo]), separator=', ')))
        print("    right end = {}".format(np.array2string(np.hstack([self.sensor_rgtarm_ep, self.sensor_rgtarm_eo]), separator=', ')))

        print("mj_data.qvel :")
        print("    base      = {}".format(np.array2string(np.hstack([self.sensor_base_linear_vel, self.sensor_base_gyro]), separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.sensor_wheel_qvel, separator=', ')))
        print("    slide     = {}".format(np.array2string(self.sensor_slide_qvel, separator=', ')))
        print("    head      = {}".format(np.array2string(self.sensor_head_qvel, separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.sensor_lft_arm_qvel, separator=', ')))
        print("    left  grp = {}".format(np.array2string(self.sensor_lft_gripper_qvel, separator=', ')))
        print("    right arm = {}".format(np.array2string(self.sensor_rgt_arm_qvel, separator=', ')))
        print("    right grp = {}".format(np.array2string(self.sensor_rgt_gripper_qvel, separator=', ')))

        print("mj_data.ctrl :")
        print("    chassis   = {}".format(np.array2string(self.mj_data.ctrl[0:2], separator=', ')))
        print("    slide     = {}".format(np.array2string(self.mj_data.ctrl[2:3], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.ctrl[3:5], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.ctrl[5:11], separator=', ')))
        print("    left  grp = {}".format(np.array2string(self.mj_data.ctrl[11:12], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.ctrl[12:18], separator=', ')))
        print("    right grp = {}".format(np.array2string(self.mj_data.ctrl[18:19], separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:]
        self.mj_data.ctrl[:self.njctrl] = self.init_joint_ctrl[:]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        self.mj_data.ctrl[:self.njctrl] = np.clip(action[:self.njctrl], self.mj_model.actuator_ctrlrange[:self.njctrl,0], self.mj_model.actuator_ctrlrange[:self.njctrl,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_qpos.tolist(),
            # "jv"   : self.sensor_qvel.tolist(),
            # "jf"   : self.sensor_force.tolist(),
            "base_position"    : self.sensor_base_position.tolist(),
            "base_orientation" : self.sensor_base_orientation.tolist(),
            "img"  : self.img_rgb_obs_s.copy(),
            "depth" : self.img_depth_obs_s.copy()
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    exec_node = MMK2Base(cfg)

    obs = exec_node.reset()
    action_list = np.zeros(19)
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action_list)
