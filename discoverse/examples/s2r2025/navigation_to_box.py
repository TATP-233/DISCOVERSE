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
        self.stm.max_state_cnt = 9
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

            # turn left
            elif self.stm.state_idx == 1:
                self.action[0] = 0.0
                self.action[1] = 0.05  # left speed
                if self.checkBaseDone(rotation=np.pi/2.0):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # forward
            elif self.stm.state_idx == 2:
                self.action[0] = 0.2  # forward speed
                self.action[1] = 0.0
                if self.checkBaseDone(translation=0.6):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()

            # turn left
            elif self.stm.state_idx == 3:
                self.action[0] = 0.0
                self.action[1] = 0.05
                if self.checkBaseDone(rotation=np.pi/2):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # forward
            elif self.stm.state_idx == 4:
                self.action[0] = 0.2
                self.action[1] = 0.0
                if self.checkBaseDone(translation=0.53):
                    self.stm.next()
                    self.action[0] = 0.0
                    self.action[1] = 0.0
                    self.initial_pose = self.get_base_pose()
                    
            # turn right a little
            elif self.stm.state_idx == 5:
                self.action[0] = 0.0
                self.action[1] = -0.01
                if self.checkBaseDone(rotation=np.pi/30):
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
            
            # stop
            elif self.stm.state_idx == 6:
                self.action[0] = 0.0
                self.action[1] = 0.0
                if self.stm.trigger():
                    self.delay_cnt = int(1.0/self.delta_t)
                else:
                    if self.checkActionDone():
                        self.stm.next()
                        self.initial_pose = self.get_base_pose()
            
            # slide down
            elif self.stm.state_idx == 7:
                self.action[2] += 2e-7 # slide down slowly
                if self.action[2] > 0.73: # down to 0.766m
                    self.stm.next()
                    self.initial_pose = self.get_base_pose()
                    
            # ACT prepare action
            elif self.stm.state_idx == 8:
                self.action = np.array([0.0, 0.0 ,0.766, 2.5958967e-04,  0.4, -4.9752708e-07,
                -1.6600089e-01,  3.2000381e-02,  5.6123108e-06,  1.5707988e+00,
                2.2230008e+00,  4.4530956e-05,  5.3182987e-07, -1.6600080e-01,
                3.2000385e-02, -5.7048383e-06, -1.5707990e+00, -2.2230008e+00,
                4.4530952e-05])
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

