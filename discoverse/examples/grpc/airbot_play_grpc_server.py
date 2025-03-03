import numpy as np
from discoverse.envs.airbot_play_base import AirbotPlayCfg, AirbotPlayBase
from grpc_server import ArrayServiceServicer, sim_serve


class AirbotPlayGrpc(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.servicer = ArrayServiceServicer(self.nj * 3, self.nj)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"
    cfg.decimation = 4
    cfg.timestep = 0.002
    cfg.use_gaussian_renderer = False

    exec_node = AirbotPlayGrpc(cfg)
    server = sim_serve(exec_node.servicer)

    while exec_node.running:
        obs, _, _, _, _ = exec_node.step(exec_node.servicer.action_array)
        exec_node.servicer.obs_array = np.concatenate(
            [[obs["time"]], obs["jq"], obs["jv"], obs["jf"]]
        )
