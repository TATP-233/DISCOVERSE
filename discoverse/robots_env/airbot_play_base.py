import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class AirbotPlayCfg(BaseConfig):
    mjcf_file_path = "mjcf/robot_airbot_play.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    init_qpos = np.zeros(7)
    obs_rgb_cam_id  = None
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {
        "arm_base"  : "airbot_play/arm_base.ply",
        "link1"     : "airbot_play/link1.ply",
        "link2"     : "airbot_play/link2.ply",
        "link3"     : "airbot_play/link3.ply",
        "link4"     : "airbot_play/link4.ply",
        "link5"     : "airbot_play/link5.ply",
        "link6"     : "airbot_play/link6.ply",
        "left"      : "airbot_play/left.ply",
        "right"     : "airbot_play/right.ply",
    }

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
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
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0
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
    cfg = AirbotPlayCfg()
    exec_node = AirbotPlayBase(cfg)

    obs = exec_node.reset()
    action = exec_node.init_joint_pose[:exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
