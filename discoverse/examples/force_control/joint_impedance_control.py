import os
import time
import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from discoverse import DISCOVERSE_ASSETS_DIR

np.set_printoptions(precision=3, suppress=True, linewidth=1000)

robot_name = "airbot_play_force"
mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", f"robot_{robot_name}.xml")

class CustomViewer:
    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):
        self.runBefore()
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            self.runFunc()
            mujoco.mj_step(self.model, self.data)
            self.sync()
            time.sleep(self.model.opt.timestep)
    
    def runBefore(self):
        pass

    def runFunc(self):
        pass

class Test(CustomViewer):
    def __init__(self, path):
        super().__init__(path, 1, azimuth=45, elevation=-30)
        self.path = path
        
    def runBefore(self):
        # 阻抗控制参数
        self.Kp = np.diag([50, 50, 50, 5, 5, 5])
        self.Kd = np.diag([5, 5, 5, 0.5, 0.5, 0.1])

        # 目标关节角度
        self.q_desired = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599]
        self.data.qpos[:self.model.nu] = self.q_desired

        # 仿真参数
        self.total_time = 30  # 总仿真时间（秒）
        self.dt = self.model.opt.timestep*2  # 仿真时间步长
        # print(self.dt)

        self.num_steps = int(self.total_time / self.dt)

        # 存储数据
        self.q_history = np.zeros((self.num_steps, self.model.nu))
        self.qdot_history = np.zeros((self.num_steps, self.model.nu))
        self.torque_history = np.zeros((self.num_steps, self.model.nu))
        self.index = 0
    
    def runFunc(self):
        # 读取当前关节角度和速度
        
        q = self.data.qpos[:(self.model.nu)]
        qdot = self.data.qvel[:(self.model.nu)]
        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 0, bias)
        
        ee_body_name = "link6"
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        R = self.data.xmat[ee_id].reshape(3, 3)

        # 创建雅可比矩阵容器
        jacp = np.zeros((3, self.model.nv))   # 末端位置雅可比
        jacr = np.zeros((3, self.model.nv))   # 末端旋转雅可比

        # 计算雅可比
        point = np.zeros(3, dtype=np.float64)
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, ee_id)
        J = np.vstack((jacp[:, :self.model.nu], jacr[:, :self.model.nu]))

        tau_act = self.data.qfrc_actuator.copy()[:self.model.nu]    # length = nv
        tau_bias = self.data.qfrc_bias.copy()[:self.model.nu]       # C + G 等偏置项，length = nv
        tau_passive = self.data.qfrc_passive.copy()[:self.model.nu] # 被动项（阻尼、弹簧），length = nv 
        Mq_full = np.zeros((self.model.nv, self.model.nv))
        qacc = self.data.qacc[:self.model.nu]
        mujoco.mj_fullM(self.model, Mq_full, self.data.qM)
        Mq = Mq_full[:self.model.nu, :self.model.nu]
        tau_mass = Mq @ qacc
        mujoco.mj_inverse(self.model, self.data)
        # tau_net = tau_act - tau_bias - tau_passive - tau_mass  # 真实由外力产生的广义力
        tau_net = self.data.qfrc_actuator[:self.model.nu] - self.data.qfrc_inverse[:self.model.nu]
        ee_ft_cal = np.linalg.pinv(J.T, rcond=1e-4) @ tau_net
        ee_ext = self.data.cfrc_ext[ee_id]
        print('cal:', ee_ft_cal)

        # 计算阻抗控制扭矩
        error = self.q_desired - q
        torque = self.Kp @ error - self.Kd @ qdot

        # 设置控制输入
        self.data.ctrl[:(self.model.nu)] = torque

        if True:
            self.q_history[self.index] = q
            self.qdot_history[self.index] = qdot
            self.torque_history[self.index] = torque
            self.index += 1

            if self.index >= self.num_steps:
                # # 绘制结果
                time = np.arange(0, self.total_time, self.dt)

                plt.figure(figsize=(12, 8))

                # 绘制关节角度
                plt.subplot(3, 1, 1)
                for j in range(self.model.nu):
                    plt.plot(time, self.q_history[:, j], label=f'Joint {j+1}')
                plt.title('Joint Angles')
                plt.xlabel('Time (s)')
                plt.ylabel('Angle (rad)')
                plt.legend()

                # 绘制关节速度
                plt.subplot(3, 1, 2)
                for j in range(self.model.nu):
                    plt.plot(time, self.qdot_history[:, j], label=f'Joint {j+1}')
                plt.title('Joint Velocities')
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (rad/s)')
                plt.legend()

                # 绘制控制扭矩
                plt.subplot(3, 1, 3)
                for j in range(self.model.nu):
                    plt.plot(time, self.torque_history[:, j], label=f'Joint {j+1}')
                plt.title('Control Torques')
                plt.xlabel('Time (s)')
                plt.ylabel('Torque (N.m)')
                plt.legend()

                plt.tight_layout()
                plt.show()

test = Test(mjcf_path)
test.run_loop()

