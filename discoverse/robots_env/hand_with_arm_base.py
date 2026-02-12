import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig

class HandWithArmCfg(BaseConfig):
    mjcf_file_path = "mjcf/inspire_hand_arm/hand_with_arm.xml"
    decimation     = 4
    timestep       = 0.001
    sync           = True
    headless       = False
    init_key       = "0"
    render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080,
    }
    obs_rgb_cam_id  = [0,1]
    use_gaussian_renderer = False #不使用高保真渲染
    
class HandWithArmBase(SimulatorBase):
    def __init__(self, config: HandWithArmCfg):
        self.nj = 18 #机器人DOF数量 机械臂 6 + 手 12 = 18
        self.na = 12 #机器人执行器数量 机械臂 6 + 手 6 = 12
        super().__init__(config)

    def post_load_mjcf(self):
        #给定初始姿态和控制
        try:
            self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
            self.init_joint_ctrl = np.zeros(self.na)
        except KeyError as e:
            self.init_joint_pose = np.zeros(self.nj)
            self.init_joint_ctrl = np.zeros(self.na)

        # TODO:补全读取触觉传感器数据
        
        self.sensor_arm_qpos = self.mj_data.sensordata[:6]
        self.sensor_arm_qvel = self.mj_data.sensordata[6:2*6]
        self.sensor_arm_force = self.mj_data.sensordata[2*6:3*6]
        self.sensor_endpoint_posi_local = self.mj_data.sensordata[3*6:3*6+3]
        self.sensor_endpoint_quat_local = self.mj_data.sensordata[3*6+3:3*6+7]
        self.sensor_endpoint_linear_vel_local = self.mj_data.sensordata[3*6+7:3*6+10]
        self.sensor_endpoint_gyro = self.mj_data.sensordata[3*6+10:3*6+13]
        self.sensor_endpoint_acc = self.mj_data.sensordata[3*6+13:3*6+16]
        
        self.sensor_finger_qpos = self.mj_data.sensordata[3*6+16:3*6+16+6] #手指主动关节的角度

    def printMessage(self):
        # TODO:终端打印必要信息
        
        # print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        # print("    arm .qpos  = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        # print("    arm .qvel  = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))
        # print("    arm .ctrl  = {}".format(np.array2string(self.mj_data.ctrl[:self.nj], separator=', ')))
        # print("    arm .force = {}".format(np.array2string(self.sensor_joint_force, separator=', ')))

        # print("    sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        # print("    sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

    def resetState(self):
        #重置状态
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.na] = self.init_joint_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        #更新控制值
        for i in range(self.na):
            #遍历赋值每个控制量
            self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(self.mj_data.ctrl[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])
 
    def checkTerminated(self):
        #判断是否终止
        return False

    def getObservation(self):
        #获取观测量
        self.obs = {
            "time" : self.mj_data.time,
            # TODO:添加触觉观测量
            "jq"   : self.sensor_arm_qpos.tolist(),
            "jv"   : self.sensor_arm_qvel.tolist(),
            "jf"   : self.sensor_arm_force.tolist(),
            "ep"   : self.sensor_endpoint_posi_local.tolist(),
            "eq"   : self.sensor_endpoint_quat_local.tolist(),
            "fq"   : self.sensor_finger_qpos.tolist(),
            "img"  : self.img_rgb_obs_s
        }
        
        #print(self.obs)
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    #当作为主程序运行时，执行这些代码
    cfg = HandWithArmCfg()
    exec_node = HandWithArmBase(cfg)

    obs = exec_node.reset()
    
    action = exec_node.init_joint_ctrl[:exec_node.na]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
