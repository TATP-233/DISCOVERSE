import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import shutil
import argparse
import multiprocessing as mp

import traceback
from discoverse.robots import AirbotPlayIK #机械臂逆运动学解算
from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSETS_DIR #引入仿真器路径和模型路径

from discoverse.utils import get_body_tmat , step_func , SimpleStateMachine #获取旋转矩阵，步进，状态机

from discoverse.robots_env.hand_with_arm_base import HandWithArmCfg #引入手臂基础配置
from discoverse.task_base.hand_arm_task_base import HandArmTaskBase , recoder_hand_with_arm 

class SimNode(HandArmTaskBase):
    #仿真节点
    def __init__(self, config: HandWithArmCfg):
        super().__init__(config)
        
    def domain_randomization(self):
        #积木位置随机化
        
        # 随机 2个绿色长方体位置
        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机 6个紫色方块位置
        for z in range(6):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            
    def check_success(self):
        #检查是否成功
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge2")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge1")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple3")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple6")
        return (
            (abs(tmat_block1[2, 2]) < 0.01)
            and (abs(abs(tmat_bridge1[1, 3] - tmat_bridge2[1, 3]) - 0.03) <= 0.002)
            and (abs(tmat_block2[2, 2]) < 0.01)
            and np.hypot(
                tmat_block1[0, 3] - tmat_block01[0, 3],
                tmat_block2[1, 3] - tmat_block02[1, 3],
            )
            < 0.11
        )

cfg = HandWithArmCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "0"
cfg.mjcf_file_path = "mjcf/inspire_hand_arm/hand_arm_bridge.xml"
cfg.obj_list = [
    "bridge1",
    "bridge2",
    "block1_green",
    "block2_green",
    "block_purple1",
    "block_purple2",
    "block_purple3",
    "block_purple4",
    "block_purple5",
    "block_purple6",
]

cfg.timestep = 0.001
cfg.sync = False
cfg.decimation = 4
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 640, "height": 480}
# cfg.obs_rgb_cam_id = [0,1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()
    
    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
        
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/build_tower")
    #如目录不存在，创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    sim_node = SimNode(cfg)
    
    if(
        hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0
    ):
        mujoco.mj_saveModel(
            sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb"))
        )
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(save_dir, os.path.basename(__file__)),
        )
    
    arm_ik = AirbotPlayIK()

    trmat = R.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))    
        
    stm = SimpleStateMachine() #有限状态机
    stm.max_state_cnt = 100 #最多状态数
    max_time = 120 #最大时间

    action = np.zeros(12) #动作空间
    process_list = []

    move_speed = 0.5
    sim_node.reset()

    #手掌旋转矩阵，使得规划后手掌掌心朝下便于抓取
    transfor = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            
        try:
            if stm.trigger():
                print(stm.state_idx)
                #print("arm_qpos is:\n",sim_node.mj_data.qpos[:6])
                if stm.state_idx == 0: #张开拇指关节
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 1: #移动到第一个绿色柱子上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block_2[:3, 3] = tmat_block_2[:3, 3] + np.array(
                        [0.035, -0.01, 0.08]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_2

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                    
                        
                elif stm.state_idx ==2 : #食指拇指抓取木块
                    for i in range(6):
                        sim_node.target_control[i] += 0
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                    
                elif stm.state_idx == 3:  # 将木块抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block_2[:3, 3] = tmat_block_2[:3, 3] + np.array(
                        [0.035, -0.01, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_2

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0] 

                elif stm.state_idx == 4:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 5:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.025, 0.1]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 6:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 7:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.025, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 8: #移动到第二个绿色柱子上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.08]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1, 0.3, 0, 0, 0, 0]
                
                elif stm.state_idx ==9 : #食指拇指抓取木块
                    for i in range(6):
                        sim_node.target_control[i] += 0
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                    
                elif stm.state_idx == 10:  # 将木块抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_block_1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block_1[:3, 3] = tmat_block_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0] 
                
                elif stm.state_idx == 11:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 12:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.1]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 13:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0, 0, 0, 0]
                    
                elif stm.state_idx == 14:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.11, -0.025, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 15:  # 手指适当弯曲便于抓取紫色立方体
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 16: #移动到第一个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 17: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==18 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.38, 0.64, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 19: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    

                elif stm.state_idx == 20:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==21:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.14]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 22:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 23:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 24: #移动到第二个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 25: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==26 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.38, 0.64, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 27: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 28:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==29:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.17]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 30:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 31:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.01, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    
                elif stm.state_idx == 32: #移动到第三个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 33: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==34 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 35: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 36:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==37:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.22]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 38:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 39:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.02, -0.02, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 40: #移动到第四个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 41: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==42 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 43: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 44:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==45:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.15]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 46:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 47:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.20]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 48: #移动到第五个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 49: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==50 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 51: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 52:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==53:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.18]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 54:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 55:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.20]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    
                elif stm.state_idx == 56: #移动到第六个紫色立方体上方
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.5, 0.5, 0.5, 0.5]
                
                elif stm.state_idx == 57: #下降
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.07]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]

                elif stm.state_idx ==58 : #食指拇指抓取木块
                    sim_node.target_control[6:] = [1.1, 0.385, 0.63, 0.4, 0.4, 0.4]
                
                elif stm.state_idx == 59: #抬起
                    trmat = R.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_cube_1 = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_cube_1[:3, 3] = tmat_cube_1[:3, 3] + np.array(
                        [0.035, -0.01, 0.2]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_cube_1

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 60:  # 将木块移动到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.25]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx ==61:  # 将木块移动下去
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.22]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                elif stm.state_idx == 62:  # 松手
                    sim_node.target_control[6:] = [1.1, 0.3, 0.4, 0.4, 0.4, 0.4]
                    
                elif stm.state_idx == 63:  # 抬起
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.108, -0.027, 0.28]
                    )
                    
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge

                    #逆运动学求解机械臂六自由度控制值    
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat@transfor, sim_node.mj_data.qpos[:6]
                    )
                
                # else :
                #     for i in range (12):
                #         sim_node.target_control[i] += 0
                #     sim_node.target_control[6:] = [1.1, 0.37, 0.6, 0, 0, 0]
                
                # for i in range(6, 12):
                #         sim_node.target_control[i] = 0
                        
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)


            
            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time Out")
            
            else:
                stm.update()
                
            if sim_node.checkActionDone():
                stm.next()
            
        except ValueError as ve :
            traceback.print_exc()
            sim_node.reset()
            
        for i in range(sim_node.na):
            action[i] = step_func(
                    action[i],
                    sim_node.target_control[i],
                    move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
                )
        
        obs, _, _, _, _ = sim_node.step(action)
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                act_lst.append(action.tolist().copy())
                obs_lst.append(obs)
                
        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(
                    target=recoder_hand_with_arm, args=(save_path, act_lst, obs_lst, cfg)
                )
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()