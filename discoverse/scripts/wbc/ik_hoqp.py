import os
import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from Task import Task
from HoQp import HoQp
from airbot_play_tasks import Abt_HOQP

from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

print_once = False


class SimNode(AirbotPlayBase):
    init_pose = np.array([0.3, 0.0, 0.22, 0.0, 0.0, 0.0])
    # init_pose = np.array([0.3, 0.0, 0.22, 0.0, 1.5708, 0.0])
    pos_ctr = True

    def resetState(self):
        super().resetState()
        self.target_pose = self.init_pose.copy()
        self.tp_se3 = pin.SE3(
            Rotation.from_euler("xyz", self.target_pose[3:]).as_matrix(),
            self.target_pose[:3],
        )

    def updateState(self):
        super().updateState()
        base_rmat = self.mj_data.body("arm_base").xmat.reshape((3, 3))
        base_posi = self.mj_data.body("arm_base").xpos
        self.mj_model.body("coordinate").pos = (
            base_rmat @ self.target_pose[:3] + base_posi
        )
        quat_xyzw = Rotation.from_matrix(
            base_rmat @ Rotation.from_euler("xyz", self.target_pose[3:]).as_matrix()
        ).as_quat()
        self.mj_model.body("coordinate").quat = quat_xyzw[[3, 0, 1, 2]]

    def getObservation(self):
        self.obs = {
            "time": self.mj_data.time,
            "jq": self.mj_data.qpos[: self.nj].tolist(),
            "endposi": (
                self.mj_data.site("endpoint").xpos - self.mj_data.body("arm_base").xpos
            )
            @ self.mj_data.body("arm_base").xmat.reshape((3, 3)),
            "endrmat": self.mj_data.site("endpoint").xmat.reshape((3, 3))
            @ self.mj_data.body("arm_base").xmat.reshape((3, 3)).T,
        }
        return self.obs

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord("t"):
            self.pos_ctr = not self.pos_ctr
        if self.pos_ctr:
            if key == ord("w"):
                self.target_pose[0] += 0.01
            elif key == ord("s"):
                self.target_pose[0] -= 0.01
            elif key == ord("a"):
                self.target_pose[1] += 0.01
            elif key == ord("d"):
                self.target_pose[1] -= 0.01
            elif key == ord("x"):
                self.target_pose[2] += 0.01
            elif key == ord("z"):
                self.target_pose[2] -= 0.01
        else:
            if key == ord("w"):
                self.target_pose[3] += 0.05
            elif key == ord("s"):
                self.target_pose[3] -= 0.05
            elif key == ord("a"):
                self.target_pose[4] += 0.05
            elif key == ord("d"):
                self.target_pose[4] -= 0.05
            elif key == ord("x"):
                self.target_pose[5] += 0.05
            elif key == ord("z"):
                self.target_pose[5] -= 0.05
        self.tp_se3 = pin.SE3(
            Rotation.from_euler("xyz", self.target_pose[3:]).as_matrix(),
            self.target_pose[:3],
        )
        return ret

    def printMessage(self):
        global print_once
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print(
            "joint q     = {}".format(
                np.array2string(self.mj_data.qpos[: self.nj], separator=", ")
            )
        )
        print("joint v     = {}".format(np.array2string(self.jv, separator=", ")))
        print(
            "end posi    = {}".format(
                np.array2string(self.obs["endposi"], separator=", ")
            )
        )
        print_once = True


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    urdf_path = os.path.join(
        DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
    )
    abt_wbc = Abt_HOQP(urdf_path)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/airbot_play_floor.xml"
    cfg.obs_rgb_cam_id = -1
    cfg.render_set = {"fps": 30, "width": 1920, "height": 1080}
    cfg.init_joint_pose = {
        "joint1": 0.0,
        "joint2": -0.0,
        "joint3": 0.0,
        "joint4": 1.5708,
        "joint5": 0.0,
        "joint6": -1.5708,
        "gripper": 0.0,
    }

    sim_node = SimNode(cfg)
    obs = sim_node.reset()
    print(obs["endposi"])

    sim_node.options.frame = mujoco.mjtFrame.mjFRAME_SITE.value

    action = sim_node.init_joint_pose[: sim_node.nj].copy()

    while sim_node.running:
        q = np.array(obs["jq"][:6]).copy()
        oMi = pin.SE3(obs["endrmat"], obs["endposi"])
        iMd = sim_node.tp_se3.actInv(oMi)
        err = pin.log(iMd).vector
        err[3:] = err[3:] @ obs["endrmat"].T
        err = np.clip(err, -1.0, 1.0).reshape((6, 1))

        jac = abt_wbc.getJac(q)
        # pin.forwardKinematics(abt_wbc.pin_model_, abt_wbc.pin_data_, q)
        # jac = pin.computeJointJacobian(abt_wbc.pin_model_, abt_wbc.pin_data_, q, 6)
        # jac = -np.dot(pin.Jlog6(iMd.inverse()), jac)

        # v = -jac.T.dot(np.linalg.solve(jac.dot(jac.T) + 1e-12 * np.eye(6), err))
        # q_new = pin.integrate(abt_wbc.pin_model_, q, v)

        # tarjv = (q_new - q)
        # print(tarjv)

        # task0 = Task(jac, -err, abt_wbc.tld_, abt_wbc.tlf_)

        task0 = Task(jac[:3], -err[:3], abt_wbc.tld_, abt_wbc.tlf_)
        task1 = Task(jac[3:], -err[3:], None, None)

        # task0 = Task(jac[3:], tar[3:], abt_wbc.tld_, abt_wbc.tlf_)
        # task1 = Task(jac[:3], tar[:3], None, None)

        hoQp0 = HoQp(task0)
        v = hoQp0.getSolutions()
        try:
            hoQp1 = HoQp(task1, hoQp0)
            v = hoQp1.getSolutions()
        except TypeError:
            print("hoQp1 failed")
            v = hoQp0.getSolutions()

        tarjv = v.T[0]

        if print_once and False:
            print("--------------------")
            print("time = {:.3f}s".format(obs["time"]))
            print("endposi =\n {}".format(obs["endposi"]))
            print("endxmat =\n {}".format(obs["endrmat"]))
            print("end rpy =\n {}".format(euler))

            print("jac   =\n{}".format(jac))
            print("tar   =\n {}".format(tar.T[0]))
            print("tarjv =\n {}".format(tarjv))
            print_once = False

        for i in range(6):
            action[i] += (
                tarjv[i] * sim_node.config.decimation * sim_node.mj_model.opt.timestep
            )
            action[i] = np.clip(action[i], obs["jq"][i] - 0.05, obs["jq"][i] + 0.05)

        obs, pri_obs, rew, ter, info = sim_node.step(action)
