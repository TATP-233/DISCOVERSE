import cv2
import numpy as np

from discoverse.utils import step_func
from discoverse.examples.robots.airbot_replay import AirbotReplay

# from block_place import SimNode, cfg
# from coffeecup_place import SimNode, cfg
# from cuplid_cover import SimNode, cfg
# from drawer_open import SimNode, cfg
# from jujube_pick import SimNode, cfg
# from jujube_place import SimNode, cfg
# from laptop_close import SimNode, cfg
from kiwi_place import SimNode, cfg

# from mouse_push import SimNode, cfg

if __name__ == "__main__":

    cfg.sync = True
    cfg.headless = False
    cfg.use_gaussian_renderer = True
    cfg.obs_rgb_cam_id = [1]
    cfg.render_set = {
        "fps": 30,
        "width": 1280,
        "height": 720,
    }
    sim_node = SimNode(cfg)
    sim_node.cam_id = -1
    sim_node.free_camera.lookat[:] = np.array([-0.05, 0.878, 0.8])
    sim_node.free_camera.distance = 0.8
    sim_node.free_camera.azimuth = 48
    sim_node.free_camera.elevation = -30

    replay = AirbotReplay("can0", with_eef=True, auto_control=True, control_period=0.05)

    move_speed = 5.0
    render_cnt = 0
    action = np.zeros(sim_node.nj)

    sim_node.reset()
    cv2.namedWindow("first_view", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("first_view", cfg.render_set["width"], cfg.render_set["height"])

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            render_cnt = 0
            action[:] = sim_node.target_control[:]

        sim_node.target_control[:] = np.array(
            [encoder.pos for encoder in replay.encoders]
        )
        for i in range(sim_node.nj - 1):
            action[i] = step_func(
                action[i], sim_node.target_control[i], move_speed * sim_node.delta_t
            )
        action[sim_node.nj - 1] = sim_node.target_control[sim_node.nj - 1]

        obs, _, _, _, _ = sim_node.step(action)

        if render_cnt < sim_node.mj_data.time * cfg.render_set["fps"]:
            cv2.imshow("first_view", cv2.cvtColor(obs["img"][1], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    cv2.destroyAllWindows()
