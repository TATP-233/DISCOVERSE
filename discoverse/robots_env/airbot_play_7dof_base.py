"""
7DOF Play G3 机械臂：与 AirbotPlayBase 结构一致，多一个关节轴 (nj=8: 7 arm + 1 gripper)
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig


class AirbotPlay7dofCfg(BaseConfig):
    mjcf_file_path = "mjcf/manipulator/roombia/airplay_7dof_pick_blocks.xml"
    # decimation     = 10   # 降低控制频率，与 airbot_play 一致更易稳定（10 * 0.001 = 0.01s/step）
    # timestep       = 0.001  # 与 MJCF option 一致
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    # 7DOF joint6/7 限位与 6DOF 不同，需在 [-0.785, 0.785] 内；joint4 在 [0, 2.618]
    init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs_rgb_cam_id  = None
    rb_link_list   = ["arm_base", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6", "Link7", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {}


class AirbotPlay7dofBase(SimulatorBase):
    def __init__(self, config: AirbotPlay7dofCfg):
        self.nj = 8
        super().__init__(config)

    def post_load_mjcf(self):
        try:
            if hasattr(self.config, "init_qpos") and self.config.init_qpos is not None:
                assert len(self.config.init_qpos) == self.nj, "init_qpos length must match the number of joints"
                self.init_joint_pose = np.array(self.config.init_qpos) # 这里有关节转换关系
                self.init_joint_ctrl = self.init_joint_pose.copy()
            else:
                raise KeyError("init_qpos not found in config")
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.nj)

        self.sensor_joint_qpos = self.mj_data.sensordata[:self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj:2*self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2*self.nj:3*self.nj]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*self.nj:3*self.nj+3]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*self.nj+3:3*self.nj+7]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*self.nj+7:3*self.nj+10]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*self.nj+10:3*self.nj+13]
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*self.nj+13:3*self.nj+16]

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

        print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # 若有 keyframe，优先用 keyframe 0（home）保证有效初始姿态
        if self.mj_model.nkey > 0:
            mujoco.mj_setKeyframe(self.mj_model, self.mj_data, 0)
            self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()  # 同步 ctrl
        else:
            self._apply_init_qpos()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _apply_init_qpos(self):
        """无 keyframe 时手动设置 qpos"""
        self.mj_data.qpos[:7] = self.init_joint_pose[:7].copy()
        self.mj_data.qpos[7] = self.init_joint_pose[7].copy()   # endright [0, 0.04]
        self.mj_data.qpos[8] = -self.init_joint_pose[7].copy()  # endleft [-0.04, 0]
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()

    def updateControl(self, action):
        self.mj_data.ctrl[:self.nj] = np.clip(action[:self.nj], self.mj_model.actuator_ctrlrange[:self.nj,0], self.mj_model.actuator_ctrlrange[:self.nj,1])

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time" : self.mj_data.time,
            "jq"   : self.sensor_joint_qpos.tolist(),
            "jv"   : self.sensor_joint_qvel.tolist(),
            "jf"   : self.sensor_joint_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "img"  : self.img_rgb_obs_s.copy(),
            "depth" : self.img_depth_obs_s.copy()
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None


if __name__ == "__main__":
    cfg = AirbotPlay7dofCfg()
    exec_node = AirbotPlay7dofBase(cfg)

    obs = exec_node.reset()
    action = exec_node.init_joint_pose[: exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
