import numpy as np
from scipy.spatial.transform import Rotation
import threading
import rclpy
from time import sleep
import subprocess

from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from task_base import MMK2TaskBase

class SimNode(MMK2TaskBase):
    def __init__(self):
        super().__init__()
        self.init_play()

    def init_play(self):
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = 15
        self.max_time = 30.0
        self.action = np.zeros_like(self.target_control)
        self.step_mode = True
        self.delta_t = 0.005 # step 5ms

    def play_once(self):
        while self.running:
            self._process_state_machine()
            self.step(self.action)

    def _process_state_machine(self):
        try:
            # confirm receive message
            if self.stm.state_idx == 0:
                if self.checkReceiveMessage():
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
            
            # ACT prepare action
            elif self.stm.state_idx == 1:
                self.action[0] = 0.0  # 前进速度
                self.action[1] = 0.0
                self.action[2:] = np.load("../../../policies/act/last_action.npy") # load from act's last action
                if self.stm.trigger():
                    self.delay_cnt = int(1.0/self.delta_t)
                else:
                    if self.checkActionDone():
                        self.stm.next()
            
            # back
            elif self.stm.state_idx == 2:
                self.action[0] = -0.1 # back speed
                self.action[1] = 0.0
                self.action[2:] = np.load("../../../policies/act/last_action.npy")
                if self.checkBaseDone(translation=0.3):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # slide up
            elif self.stm.state_idx == 3:
                self.action[0] = 0.0
                self.action[1] = 0.0
                self.action[2] -= 1e-7 # slide up slowly
                if self.action[2] < 0.01:
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()

            # turn left
            elif self.stm.state_idx == 4:
                self.action[0] = 0.0
                self.action[1] = 0.05  # left speed
                if self.checkBaseDone(rotation=np.pi/2):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # forward
            elif self.stm.state_idx == 5:
                self.action[0] = 0.2
                self.action[1] = 0.0
                if self.checkBaseDone(translation=0.5):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # turn left
            elif self.stm.state_idx == 6:
                self.action[0] = 0.0
                self.action[1] = 0.05
                if self.checkBaseDone(rotation=np.pi/2-np.pi/20):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # forward
            elif self.stm.state_idx == 7:
                self.action[0] = 0.2
                self.action[1] = 0.0
                if self.checkBaseDone(translation=0.3):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # turn left a little
            elif self.stm.state_idx == 8:
                self.action[0] = 0.0
                self.action[1] = 0.01
                if self.checkBaseDone(rotation=np.pi/40):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # stop
            elif self.stm.state_idx == 9:
                self.action[0] = 0.0
                self.action[1] = 0.0
                if self.stm.trigger():
                    self.delay_cnt = int(1.0/self.delta_t)
                else:
                    if self.checkActionDone():
                        self.stm.next()
                        self.initial_pose = self.get_base_pose()
            
            # slide down
            elif self.stm.state_idx == 10:
                self.action[2] += 1e-7 # slide down slowly
                if self.action[2] > 0.45:
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
            
            # loose box
            elif self.stm.state_idx == 11:
                self.action[9] -= 1e-7 # left joint5
                self.action[16] += 1e-7 # right joint5
                if self.action[9] < -0.8 and self.action[16] > 1.2:
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # slide up
            elif self.stm.state_idx == 12:
                self.action[0] = 0.0
                self.action[1] = 0.0
                self.action[2] -= 1e-7
                if self.action[2] < 0.01:
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # ACT prepare action
            elif self.stm.state_idx == 13:
                self.action = np.array([0.0, 0.0 ,3.3936809e-05,  1.7894521e-04,  0.0, -5.4910754e-05,
                                        -1.6599999e-01,  3.2000028e-02,  8.5838783e-06,  1.5708048e+00,
                                        6.5219975e-01,  4.4530989e-05,  5.4914592e-05, -1.6599984e-01,
                                        3.2000016e-02, -8.9520818e-06, -1.5708050e+00, -6.5219980e-01,
                                        4.4530967e-05])
                if self.stm.trigger():
                    self.delay_cnt = int(0.1/self.delta_t)
                else:
                    if self.checkActionDone():
                        self.stm.next()
                        self.running=False # done navigation

        except ValueError as ve:
            print(f"[ERROR] {ve}")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    rclpy.init()
    mmk2_node = SimNode()
    spin_thead = threading.Thread(target=lambda: rclpy.spin(mmk2_node))
    spin_thead.start()
    pub_thread = threading.Thread(target=mmk2_node.pub_thread)
    pub_thread.start()

    mmk2_node.play_once()
    # spin_thead.join()
    # pub_thread.join()
    mmk2_node.destroy_node()
    rclpy.shutdown()

