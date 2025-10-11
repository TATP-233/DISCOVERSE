import numpy as np
from dummy_robot import DummyRobot, DummyRobotConfig
import mujoco
class MMK2SlamCfg(DummyRobotConfig):
    robot_name     = "mmk2"
    mjcf_file_path = "mjcf/mmk2_car.xml"
    timestep       = 0.0025
    decimation     = 4
    rb_link_list   = [
        "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
        "lft_arm_base", "lft_arm_link1", "lft_arm_link2", 
        "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "lft_finger_left_link", "lft_finger_right_link", 
        "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2", 
        "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link"
    ]
    gs_model_dict  = {
        "agv_link"              :   "mmk2/agv_link.ply",
        "slide_link"            :   "mmk2/slide_link.ply",
        "head_pitch_link"       :   "mmk2/head_pitch_link.ply",
        "head_yaw_link"         :   "mmk2/head_yaw_link.ply",

        "lft_arm_base"          :   "airbot_play/arm_base.ply",
        "lft_arm_link1"         :   "airbot_play/link1.ply",
        "lft_arm_link2"         :   "airbot_play/link2.ply",
        "lft_arm_link3"         :   "airbot_play/link3.ply",
        "lft_arm_link4"         :   "airbot_play/link4.ply",
        "lft_arm_link5"         :   "airbot_play/link5.ply",
        "lft_arm_link6"         :   "airbot_play/link6.ply",
        "lft_finger_left_link"  :   "airbot_play/left.ply",
        "lft_finger_right_link" :   "airbot_play/right.ply",

        "rgt_arm_base"          :   "airbot_play/arm_base.ply",
        "rgt_arm_link1"         :   "airbot_play/link1.ply",
        "rgt_arm_link2"         :   "airbot_play/link2.ply",
        "rgt_arm_link3"         :   "airbot_play/link3.ply",
        "rgt_arm_link4"         :   "airbot_play/link4.ply",
        "rgt_arm_link5"         :   "airbot_play/link5.ply",
        "rgt_arm_link6"         :   "airbot_play/link6.ply",
        "rgt_finger_left_link"  :   "airbot_play/left.ply",
        "rgt_finger_right_link" :   "airbot_play/right.ply"
    }

class MMK2SlamRobot(DummyRobot):
    pitch_joint_id = 4

    def move_camera_pitch(self, d_pitch):
        self.mj_data.ctrl[self.pitch_joint_id] += d_pitch

if __name__ == "__main__":
    cfg = MMK2SlamCfg()
    cfg.y_up = False
    
    robot_cam_id = 0
    cfg.obs_rgb_cam_id   = [robot_cam_id]
    cfg.obs_depth_cam_id = [robot_cam_id]

    cfg.render_set["fps"] = 60
    cfg.render_set["width"] = 1920
    cfg.render_set["height"] = 1080
    cfg.mjcf_file_path = "mjcf/mmk2_car.xml"

    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "scene/car/lixiang.ply"
    cfg.gs_model_dict["Box2"] = "scene/car/hougai.ply"
    cfg.gs_model_dict["background_env"] = "scene/car/environment.ply"
    cfg.obj_list     = ["Box2"]
    robot = MMK2SlamRobot(cfg)
    robot.cam_id = robot_cam_id

    action = np.zeros(4)
    # if z_up:
        # action[0] : lineal_velocity_x  local m    不论yup还是zup，始终为朝前为正方向
        # action[1] : lineal_velocity_y  local m    不论yup还是zup，始终为朝左为正方向
        # action[2] : angular_velocity_z rad        不论yup还是zup，始终为从上向下看逆时针旋转为正方向
        # action[3] : camera_pitch       rad        不论yup还是zup，始终为镜头俯仰
    # elif y_up:
        # action[0] : lineal_velocity_x   local m
        # action[1] : lineal_velocity_-z  local m
        # action[2] : angular_velocity_y  rad
        # action[3] : camera_pitch        rad

    obs = robot.reset()
    rgb_cam_posi = obs["rgb_cam_posi"]
    depth_cam_posi = obs["depth_cam_posi"]
    rgb_img_0 = obs["rgb_img"][0]
    depth_img_0 = obs["depth_img"][0]

    print("rgb_cam_posi    = ", rgb_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("depth_cam_posi  = ", depth_cam_posi)
    # [[posi_x, posi_y, posi_z], [quat_w, quat_x, quat_y, quat_z]]
    # [(array([0., 0., 1.]), array([ 0.49999816,  0.50000184, -0.5       , -0.5       ]))]

    print("rgb_img.shape   = ", rgb_img_0.shape  , "rgb_img.dtype    = ", rgb_img_0.dtype)
    # rgb_img.shape   =  (1080, 1920, 3) rgb_img.dtype    =  uint8

    print("depth_img.shape = ", depth_img_0.shape, "depth_img.dtype  = ", depth_img_0.dtype)
    # depth_img.shape =  (1080, 1920, 1) depth_img.dtype  =  float32

    robot.printHelp()

    while robot.running:
        obs, _, _, _, _ = robot.step(action)
        # robot.key_board_callback(kb_listener)
        # robot.step()
        pos, _ = robot.getObjPose("mmk2")
        if -1 < pos[0] < 2 :
            robot.mj_data.ctrl[mujoco.mj_name2id(robot.mj_model,  mujoco.mjtObj.mjOBJ_ACTUATOR, "trunk_lid_open")] = float(0.2)
        else:
            robot.mj_data.ctrl[mujoco.mj_name2id(robot.mj_model,  mujoco.mjtObj.mjOBJ_ACTUATOR, "trunk_lid_open")] = float(-0.9)