import os
import mujoco
import mediapy
import numpy as np

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class CARCfg(BaseConfig):
    mjcf_file_path = "mjcf/rm2_car_floor.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080 
    }
    obs_rgb_cam_id   = None
    obs_depth_cam_id = None

    use_gaussian_renderer = False
    gs_model_dict  = {
        "rm2"               :   "rm2_car/library/rm2.ply",
        "wheel1"            :   "rm2_car/library/rgt_wheel.ply",
        "wheel2"            :   "rm2_car/library/lft_wheel.ply",
        "wheel3"            :   "rm2_car/library/lft_wheel.ply",
        "wheel4"            :   "rm2_car/library/rgt_wheel.ply",
        "steering1"         :   "rm2_car/library/steering_left_front.ply",
        "steering2"         :   "rm2_car/library/steering_right_front.ply",
        "steering3"         :   "rm2_car/library/steering_left_back.ply",
        "steering4"         :   "rm2_car/library/steering_right_back.ply",

        "arm_base"          :   "airbot_play/arm_base.ply",
        "arm_link1"         :   "airbot_play/link1.ply",
        "arm_link2"         :   "airbot_play/link2.ply",
        "arm_link3"         :   "airbot_play/link3.ply",
        "arm_link4"         :   "airbot_play/link4.ply",
        "arm_link5"         :   "airbot_play/link5.ply",
        "arm_link6"         :   "airbot_play/link6.ply",
        "finger_left_link"  :   "airbot_play/left.ply",
        "finger_right_link" :   "airbot_play/right.ply",

    }
    """
    njqpos=22
    [0:7]-base; 
    7-steer_joint1 ;  8-wheel_joint1;
    9-steer_joint2 ; 10-wheel_joint2;
    11-steer_joint3; 12-wheel_joint3;
    13-steer_joint4; 14-wheel_joint4;
    [15-20]-arm_joint1 to arm_joint6
    21 gripper_joint

    njctrl=13
    0-forward; 1-turn; 
    2-steer_joint1; 3-steer_joint2; 4-steer_joint3; 5-steer_joint4;
    6-11 arm_joint1 to arm_joint6
    12 gripper_joint
    """

class CARBase(SimulatorBase):
    def __init__(self, config: CARCfg):
        self.njq = 22
        self.njv = 21
        self.njctrl = 13

        super().__init__(config)
        self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.njq]

        ip_cp = self.init_joint_pose.copy()
        self.init_ctrl = np.array(
            [0.0, 0.0] +  # [0:2] 控制前进（forward）和转向（turn）
            ip_cp[[7,9,11,13]].tolist() +  # 控制转向关节（steer_joint）和轮子（wheel_joint）
            ip_cp[15:22].tolist()   # [15:22] 控制机械臂的关节
        )

        self.resetState()

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:self.njq].copy()
        self.mj_data.ctrl[:self.njctrl] = self.init_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        varr = np.hypot(action[0], action[1])
        if abs(action[2]) > 1e-3:
            self.mj_data.ctrl[0] = 0.0
            self.mj_data.ctrl[1] = action[2]
            self.mj_data.ctrl[2] = -np.pi/4.
            self.mj_data.ctrl[3] =  np.pi/4.
            self.mj_data.ctrl[4] = -np.pi/4.
            self.mj_data.ctrl[5] =  np.pi/4.
        elif varr > 1e-3:
            self.mj_data.ctrl[1] = 0.0
            if action[0] > 0.0:
                self.mj_data.ctrl[0] = varr
                v_dir = np.arctan2(action[1], action[0])
            else:
                self.mj_data.ctrl[0] = -varr
                if action[1] > 0.0:
                    v_dir = -np.pi + np.arctan2(action[1], action[0])
                else:
                    v_dir = np.pi + np.arctan2(action[1], action[0])
            self.mj_data.ctrl[2] = v_dir
            self.mj_data.ctrl[3] = v_dir
            self.mj_data.ctrl[4] = v_dir
            self.mj_data.ctrl[5] = v_dir
        else:
            self.mj_data.ctrl[0] = 0.0
            self.mj_data.ctrl[1] = 0.0

        # arm joint control
        for i in range(6):
            self.mj_data.ctrl[6+i] = action[3+i]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"    : self.mj_data.qpos[:self.njq].tolist(),
            "jv"    : self.mj_data.qvel[:self.njv].tolist(),
            "img"   : self.img_rgb_obs_s,
            "depth" : self.img_depth_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

    def teleopProcess(self):
        global action

        action[0] =  self.teleop.joy_cmd.axes[1]*20 * abs(self.teleop.joy_cmd.axes[1])
        action[1] =  self.teleop.joy_cmd.axes[0]*20 * abs(self.teleop.joy_cmd.axes[0])
        action[2] = -self.teleop.joy_cmd.axes[3]*10 * abs(self.teleop.joy_cmd.axes[3])

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=500)

    cfg = CARCfg()
    exec_node = CARBase(cfg)
    obs = exec_node.reset()

    action = np.zeros(10)
    # action[0] : lineal_velocity_x  local    朝前为正方向
    # action[1] : lineal_velocity_y  local    朝左为正方向
    # action[2] : angular_velocity_z          为从上向下看逆时针旋转为正方向

    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)