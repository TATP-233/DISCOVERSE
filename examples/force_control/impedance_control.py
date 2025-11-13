import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
import time
from collections import deque

class ImpedanceController:
    def __init__(self, mj_model, kpl, kdl, enable_plot=False, max_points=100):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.Kp = np.diag(kpl)
        self.Kd = np.diag(kdl)
        self.ndof = len(kpl)
        self.q_desired = np.zeros(self.ndof)
        self.bias = np.zeros(mj_model.nv)

        self.enable_plot = enable_plot
        self.enable_plot = enable_plot
        if enable_plot:
            self.max_points = max_points
            # 使用多进程队列传递数据
            self.plot_queue = mp.Queue(maxsize=100)
            self.plot_process = None
            self.start_plot_process()


    def set_target(self, q_desired):
        self.q_desired = q_desired

    def update_state(self, q, dq, tau=None): # modify
        self.q = q.copy()
        self.dq = dq.copy()
        self.tau = tau.copy() if tau is not None else None  
        self.mj_data.qpos[:self.ndof] = q.copy()
        self.mj_data.qvel[:self.ndof] = dq.copy()
        if tau is not None:
            self.mj_data.ctrl[:len(tau)] = tau.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def get_ext_force(self): # modify
        ee_body_name = "link6"
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        tau_total = self.mj_data.qfrc_inverse[:self.ndof].copy() # Total Toque
        self.mj_data.qvel[:] = np.zeros(self.ndof)
        self.mj_data.qacc[:] = np.zeros(self.ndof)
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        tau_gravity = self.mj_data.qfrc_inverse[:self.ndof].copy() # Gravity Toque
        jacp = np.zeros((3, self.mj_model.nv))   
        jacr = np.zeros((3, self.mj_model.nv))   
        point = np.zeros(3, dtype=np.float64)
        mujoco.mj_jac(self.mj_model, self.mj_data, jacp, jacr, point, ee_id)
        J = np.vstack((jacp[:, :self.mj_model.nu], jacr[:, :self.mj_model.nu])) # Jacobian Matrix
        force_ext = np.linalg.pinv(J.T) @ (tau_gravity - tau_total)
        # print(force_ext)
        return force_ext

    def compute_torque(self):      
        position_error = self.q_desired - self.q
        velocity_error = -self.dq
        mujoco.mj_rne(self.mj_model, self.mj_data, 0, self.bias)
        torque = self.Kp @ position_error + self.Kd @ velocity_error + self.bias[:self.mj_model.nu]
        return torque
    
    # ---------------- 绘图相关 ----------------
    def start_plot_process(self):
        """启动绘图进程"""
        if self.plot_process is None or not self.plot_process.is_alive():
            self.plot_process = mp.Process(
                target=plot_worker, 
                args=(self.plot_queue, self.max_points),
                daemon=True
            )
            self.plot_process.start()
    
    def _update_plot_data(self, force_ext, t):
        """更新绘图数据并通过队列发送"""
        if not self.enable_plot:
            return
            
        # 准备要发送的数据
        plot_data = {
            'force': force_ext[:3],
            'torque': force_ext[3:],
            'time': t
        }
        
        # 非阻塞方式发送数据
        try:
            self.plot_queue.put_nowait(plot_data)
        except:
            # 如果队列已满，丢弃最旧的数据
            try:
                self.plot_queue.get_nowait()  # 丢弃一个旧数据
                self.plot_queue.put_nowait(plot_data)  # 放入新数据
            except:
                pass  # 如果仍然失败，跳过此次更新
    
    def update_plot(self):
        """空方法，因为绘图在独立进程中"""
        pass
    
    def run_step(self, dt=0.01):
        """
        单步更新，通过队列发送数据到绘图进程
        """
        self.update_state(self.q, self.dq, self.tau)
        force_ext = self.get_ext_force()
        self._update_plot_data(force_ext, dt)
        return force_ext
    
    def close(self):
        """清理资源"""
        if self.enable_plot and self.plot_process:
            self.plot_process.terminate()
            self.plot_process.join()

def plot_worker(queue, max_points):
    """独立的绘图进程函数"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 初始化数据存储
    force_history = deque(maxlen=max_points)
    torque_history = deque(maxlen=max_points)
    time_history = deque(maxlen=max_points)
    
    # 创建图形
    plt.ion()  # 交互模式
    fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(8, 6))
    
    # 创建线条对象
    lines_force = [
        ax_force.plot([], [], label=lbl)[0] for lbl in ["Fx", "Fy", "Fz"]
    ]
    lines_torque = [
        ax_torque.plot([], [], label=lbl)[0] for lbl in ["Mx", "My", "Mz"]
    ]
    
    # 设置坐标轴
    ax_force.set_title("External Force (N)")
    ax_force.set_xlim(0, max_points)
    ax_force.set_ylim(-50, 50)
    ax_force.legend()
    ax_force.grid(True)

    ax_torque.set_title("External Torque (Nm)")
    ax_torque.set_xlim(0, max_points)
    ax_torque.set_ylim(-10, 10)
    ax_torque.legend()
    ax_torque.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 主循环
    try:
        while True:
            # 非阻塞方式获取数据
            try:
                data = queue.get_nowait()
                
                # 更新数据
                force_history.append(data['force'])
                torque_history.append(data['torque'])
                time_history.append(data['time'])
                
                # 更新图形
                if force_history:
                    force_array = np.array(force_history)
                    torque_array = np.array(torque_history)
                    
                    for i in range(3):
                        lines_force[i].set_data(range(len(force_history)), force_array[:, i])
                        lines_torque[i].set_data(range(len(torque_history)), torque_array[:, i])
                    
                    ax_force.set_xlim(0, len(force_history))
                    ax_torque.set_xlim(0, len(torque_history))
                    
                    # 动态调整Y轴范围
                    if len(force_history) > 1:
                        force_margin = max(5, np.max(np.abs(force_array)) * 0.1)
                        ax_force.set_ylim(-np.max(np.abs(force_array)) - force_margin, 
                                         np.max(np.abs(force_array)) + force_margin)
                        
                        torque_margin = max(1, np.max(np.abs(torque_array)) * 0.1)
                        ax_torque.set_ylim(-np.max(np.abs(torque_array)) - torque_margin, 
                                          np.max(np.abs(torque_array)) + torque_margin)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            except:
                # 队列为空，等待一段时间
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        plt.close('all')
    except Exception as e:
        print(f"Plot process error: {e}")
        plt.close('all')


if __name__ == "__main__":
    import os
    from discoverse import DISCOVERSE_ASSETS_DIR

    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator", "robot_airbot_play_force.xml")
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    kp = [100, 100, 100, 5, 100, 5]
    kd = [5, 5, 5, 0.5, 0.5, 0.1]
    controller = ImpedanceController(model, kpl=kp, kdl=kd)

    controller.set_target(np.zeros(6))
    controller.update_state(q=np.zeros(6), dq=np.zeros(6), tau=np.zeros(6))
    tau = controller.compute_torque()
    print(tau)