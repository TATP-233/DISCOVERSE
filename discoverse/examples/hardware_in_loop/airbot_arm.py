import enum
import mujoco
import numpy as np

from discoverse.envs import SimulatorBase
from discoverse.utils import BaseConfig, PIDController, PIDarray


class ControlMode(enum.Enum):
    POSITION = 0
    MIT = 1


class AirbotArm(SimulatorBase):
    def __init__(self, config: BaseConfig):

        if config.eef_type == "none":
            config.mjcf_file_path = "mjcf/hardware_in_loop/airbot_play_short.xml"
        elif (
            config.eef_type == "G2"
            or config.eef_type == "E2B"
            or config.eef_type == "PE2"
        ):
            config.mjcf_file_path = (
                f"mjcf/hardware_in_loop/airbot_play_short_{config.eef_type}.xml"
            )
        else:
            raise NotImplementedError

        if config.eef_type == "none":
            self.nj = 6
            self.arm_pids = PIDarray(
                kps=np.array([225.0, 275.0, 350.0, 1.5, 2.50, 1.00]),
                kis=np.array([5.0, 25.0, 35.0, 2.5, 2.50, 2.50]),
                kds=np.array([3.0, 5.0, 7.0, 0.1, 0.25, 0.01]),
                integrator_maxs=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            )
        elif config.eef_type == "G2":
            self.nj = 7
            self.arm_pids = PIDarray(
                kps=np.array([225.0, 275.0, 350.0, 25.00, 25.00, 5.00]),
                kis=np.array([5.0, 25.0, 35.0, 2.50, 2.50, 2.50]),
                kds=np.array([3.0, 5.0, 7.0, 0.01, 0.01, 0.01]),
                integrator_maxs=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            )
            self.gripper_pid = PIDController(1.0, 0.5, 0.1, integrator_max=0.1)
            self.gripper_control_mode = ControlMode.MIT
        elif config.eef_type == "E2B" or config.eef_type == "PE2":
            self.nj = 7
            self.arm_pids = PIDarray(
                kps=np.array([225.0, 275.0, 350.0, 25.00, 25.00, 5.00]),
                kis=np.array([5.0, 25.0, 35.0, 2.50, 2.50, 2.50]),
                kds=np.array([3.0, 5.0, 7.0, 0.01, 0.01, 0.01]),
                integrator_maxs=np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            )
            self.gripper_pid = PIDController(0.0, 0.0, 0.0, integrator_max=0)
            self.gripper_control_mode = ControlMode.MIT

        self.arm_control_mode = ControlMode.POSITION
        self.narm = 6

        self.action = np.zeros(self.nj * 3)
        # self.action[:self.nj] = target_position
        # self.action[self.nj:2*self.nj] = target_velocity
        # self.action[2*self.nj:3*self.nj] = target_torque
        super().__init__(config)

    def post_load_mjcf(self):
        self.sensor_joint_qpos = self.mj_data.sensordata[: self.nj]
        self.sensor_joint_qvel = self.mj_data.sensordata[self.nj : 2 * self.nj]
        self.sensor_joint_force = self.mj_data.sensordata[2 * self.nj : 3 * self.nj]
        self.ctr_position = self.mj_data.ctrl[: self.nj]
        self.ctr_velcity = self.mj_data.ctrl[self.nj : 2 * self.nj]
        self.ctr_torque = self.mj_data.ctrl[2 * self.nj : 3 * self.nj]

        self.mit_kps = self.mj_model.actuator_gainprm[: self.nj, 0].copy()
        self.mit_kds = self.mj_model.actuator_gainprm[self.nj : self.nj * 2, 0].copy()

        self.switch_arm_control_mode(self.arm_control_mode)

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time  = {:.3f}".format(self.mj_data.time))
        print(
            "    arm .qpos  = {}".format(
                np.array2string(self.sensor_joint_qpos, separator=", ")
            )
        )
        print(
            "    arm .qvel  = {}".format(
                np.array2string(self.sensor_joint_qvel, separator=", ")
            )
        )
        print(
            "    arm .force = {}".format(
                np.array2string(self.sensor_joint_force, separator=", ")
            )
        )
        print(
            "    arm .ctrl q= {}".format(
                np.array2string(self.mj_data.ctrl[: self.nj], separator=", ")
            )
        )
        print(
            "    arm .ctrl v= {}".format(
                np.array2string(
                    self.mj_data.ctrl[self.nj * 1 : self.nj * 2], separator=", "
                )
            )
        )
        print(
            "    arm .ctrl f= {}".format(
                np.array2string(
                    self.mj_data.ctrl[self.nj * 2 : self.nj * 3], separator=", "
                )
            )
        )
        # print("    actuator_gainprm =\n{}".format(self.mj_model.actuator_gainprm))
        # print("    actuator_biasprm =\n{}".format(self.mj_model.actuator_biasprm))

    def setArmMITKPKD(self, kps, kds):
        assert (type(kps) is float and type(kds) is float) or (
            len(kps) == self.narm and len(kds) == self.narm
        )
        self.mj_model.actuator_gainprm[: self.narm, 0] = kps
        self.mj_model.actuator_biasprm[: self.narm, 1] = -kps
        self.mj_model.actuator_gainprm[self.nj : self.nj + self.narm, 0] = kds
        self.mj_model.actuator_biasprm[self.nj : self.nj + self.narm, 2] = -kds

    def setGripperMITKPKD(self, kp, kd):
        if hasattr(self, "gripper_control_mode"):
            self.mj_model.actuator_gainprm[self.narm, 0] = kp
            self.mj_model.actuator_biasprm[self.narm, 1] = -kp
            self.mj_model.actuator_gainprm[self.nj + self.narm, 0] = kd
            self.mj_model.actuator_biasprm[self.nj + self.narm, 2] = -kd

    def switch_arm_control_mode(self, target_control_mode):
        if target_control_mode == ControlMode.POSITION:
            self.action[: self.narm] = self.sensor_joint_qpos[: self.narm]
            self.arm_pids.reset()
            self.setArmMITKPKD(0.0, 0.0)
        elif target_control_mode == ControlMode.MIT:
            self.setArmMITKPKD(self.mit_kps[: self.narm], self.mit_kds[: self.narm])
        else:
            raise NotImplementedError(
                "Invalid control mode {}".format(target_control_mode)
            )
        self.arm_control_mode = target_control_mode
        print("Arm control mode: {}".format(self.arm_control_mode))

    def switch_gripper_control_mode(self, target_control_mode):
        if hasattr(self, "gripper_control_mode"):
            if target_control_mode == ControlMode.POSITION:
                self.action[self.narm] = self.sensor_joint_qpos[self.narm]
                self.gripper_pid.reset()
                self.setGripperMITKPKD(0.0, 0.0)
            elif target_control_mode == ControlMode.MIT:
                self.setGripperMITKPKD(self.mit_kps[self.narm], self.mit_kds[self.narm])
            else:
                raise NotImplementedError(
                    "Invalid control mode {}".format(target_control_mode)
                )
            self.gripper_control_mode = target_control_mode
            print("Gripper control mode: {}".format(self.gripper_control_mode))

    def windowKeyPressCallback(self, key):
        if key == ord("M") or key == ord("m"):
            self.switch_arm_control_mode(
                ControlMode((self.arm_control_mode.value + 1) % len(ControlMode))
            )
        elif key == ord("N") or key == ord("n"):
            self.switch_gripper_control_mode(
                ControlMode((self.gripper_control_mode.value + 1) % len(ControlMode))
            )

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, _):
        if self.arm_control_mode == ControlMode.POSITION:
            self.ctr_torque[: self.narm] = self.arm_pids.output(
                self.action[: self.narm] - self.sensor_joint_qpos[: self.narm],
                self.mj_model.opt.timestep,
            )
        elif self.arm_control_mode == ControlMode.MIT:
            self.ctr_position[: self.narm] = self.action[: self.narm]
            self.ctr_velcity[: self.narm] = self.action[self.nj : self.nj + self.narm]
            self.ctr_torque[: self.narm] = self.action[
                2 * self.nj : 2 * self.nj + self.narm
            ]
        else:
            raise NotImplementedError(
                "Invalid control mode {}".format(self.arm_control_mode)
            )
        if hasattr(self, "gripper_control_mode"):
            if self.gripper_control_mode == ControlMode.POSITION:
                self.ctr_torque[self.narm] = self.gripper_pid.output(
                    self.action[self.narm] - self.sensor_joint_qpos[self.narm],
                    self.mj_model.opt.timestep,
                )
            elif self.gripper_control_mode == ControlMode.MIT:
                self.ctr_position[self.narm] = self.action[self.narm]
                self.ctr_velcity[self.narm] = self.action[self.nj + self.narm]
                self.ctr_torque[self.narm] = self.action[2 * self.nj + self.narm]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time": self.mj_data.time,
            "jq": self.sensor_joint_qpos.tolist(),
            "jv": self.sensor_joint_qvel.tolist(),
            "jf": self.sensor_joint_force.tolist(),
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None


cfg = BaseConfig()
cfg.decimation = 4
cfg.timestep = 0.001
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 30, "width": 1280, "height": 720}

if __name__ == "__main__":
    import time
    import argparse

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser(
        description="Run arm with specified parameters. \ne.g. python3 airbot_play_short.py --arm_type play_short --eef_type none"
    )
    parser.add_argument(
        "--arm_type",
        type=str,
        choices=["play_long", "play_short", "lite", "pro", "replay"],
        help="Name of the arm",
        default="play_short",
    )
    # :TODO: play_long, lite, pro, replay
    parser.add_argument(
        "--eef_type",
        type=str,
        choices=["G2", "E2B", "PE2", "none"],
        help="Name of the eef",
        default="none",
    )
    # :TODO: PE2
    parser.add_argument(
        "--discoverse_viewer", action="store_true", help="Use discoverse viewer"
    )
    args = parser.parse_args()

    cfg.arm_type = args.arm_type
    cfg.eef_type = args.eef_type
    if not args.discoverse_viewer:
        import mujoco.viewer

        cfg.enable_render = False

    exec_node = AirbotArm(cfg)

    obs = exec_node.reset()

    def func_while_running():
        exec_node.action[: exec_node.nj] = 0.15
        obs, pri_obs, rew, ter, info = exec_node.step()

    if args.discoverse_viewer:
        while exec_node.running:
            step_start = time.time()
            func_while_running()
            time_until_next_step = exec_node.delta_t - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    else:
        with mujoco.viewer.launch_passive(
            exec_node.mj_model,
            exec_node.mj_data,
            key_callback=exec_node.windowKeyPressCallback,
        ) as viewer:
            while viewer.is_running():
                step_start = time.time()
                func_while_running()
                viewer.sync()
                time_until_next_step = exec_node.delta_t - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
