import numpy as np
import mujoco
from discoverse.envs.simulator import SimulatorBase


class Simulator(SimulatorBase):
    """
    奇异果任务的自定义模拟器，继承自SimulatorBase
    实现了getObservation方法以避免NotImplementedError
    """
    
    def __init__(self, cfg):
        super(Simulator, self).__init__(cfg)
        
    def getObservation(self):
        """
        获取观察值，这是一个必须实现的方法
        
        Returns:
            observation: 观察值，可以是任何形式，这里我们返回一个空字典
        """
        # 这个方法只在reset()时被调用，我们在KiwiEnv中不使用这个返回值
        # 因为我们在KiwiEnv.reset()中直接调用_get_obs()获取观察值
        return {}
