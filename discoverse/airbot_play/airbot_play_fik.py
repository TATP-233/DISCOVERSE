import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


class AirbotPlayFIK:
    bias = np.array([0.0, -2.7549, 2.7549, 1.5708, 0.0, 0.0])
    a1 = 0.1172
    a3 = 0.27009
    a4 = 0.29015
    a6 = 0.23645
    arm_joint_range = np.array(
        [[-3.09, -2.92, -0.04, -2.95, -1.8, -2.90], [2.04, 0.12, 3.09, 2.95, 1.8, 2.90]]
    )
    joint_range_scale = arm_joint_range[1] - arm_joint_range[0]

    arm_rot_mat = np.array([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def __init__(self, urdf) -> None:
        self.pin_model = pin.buildModelFromUrdf(urdf)
        self.pin_data = self.pin_model.createData()

    def forwardKin(self, q) -> pin.SE3:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        return self.pin_data.oMi[6]

    def properIK(self, pos, ori, ref_q=None):
        return self.inverseKin(pos, ori @ self.arm_rot_mat, ref_q)

    def properFK(self, q) -> pin.SE3:
        eoMi = self.forwardKin(np.array(q))
        tmat = np.eye(4)
        tmat[:3, :3] = eoMi.rotation @ self.arm_rot_mat.T
        tmat[:3, 3] = eoMi.translation.T
        return tmat

    def inverseKin(self, pos, ori, ref_q=None):
        assert len(pos) == 3 and ori.shape == (3, 3)
        pos = self.move_joint6_2_joint5(pos, ori)
        angle = [0.0] * 6
        ret = []

        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            c3 = (
                pos[0] ** 2
                + pos[1] ** 2
                + (pos[2] - self.a1) ** 2
                - self.a3**2
                - self.a4**2
            ) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError(
                    "Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori)
                )

            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3**2)
                angle[2] = np.arctan2(s3, c3)
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                angle[1] = np.arctan2(
                    k1 * (pos[2] - self.a1)
                    - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                    i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                    + k2 * (pos[2] - self.a1),
                )
                R = np.array(
                    [
                        [
                            np.cos(angle[0]) * np.cos(angle[1] + angle[2]),
                            -np.cos(angle[0]) * np.sin(angle[1] + angle[2]),
                            np.sin(angle[0]),
                        ],
                        [
                            np.sin(angle[0]) * np.cos(angle[1] + angle[2]),
                            -np.sin(angle[0]) * np.sin(angle[1] + angle[2]),
                            -np.cos(angle[0]),
                        ],
                        [np.sin(angle[1] + angle[2]), np.cos(angle[1] + angle[2]), 0],
                    ]
                )
                ori1 = R.T @ ori
                for i5 in [1, -1]:
                    angle[3] = np.arctan2(i5 * ori1[2, 2], i5 * ori1[1, 2])
                    angle[4] = np.arctan2(
                        i5 * np.sqrt(ori1[2, 2] ** 2 + ori1[1, 2] ** 2), ori1[0, 2]
                    )
                    angle[5] = np.arctan2(-i5 * ori1[0, 0], -i5 * ori1[0, 1])
                    js = self.add_bias(angle)
                    if np.all(
                        (js > self.arm_joint_range[0]) * (js < self.arm_joint_range[1])
                    ):
                        ret.append(js)
        if len(ret) == 0:
            raise ValueError(
                "Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori)
            )

        if ref_q is not None:
            joint_dist_lst = []
            for js in ret:
                joint_dist_lst.append(
                    np.sum(np.abs(ref_q - js) / self.joint_range_scale)
                )
            q = ret[np.argmin(joint_dist_lst)]
            return q
        else:
            return ret

    def add_bias(self, angle):
        ret = []
        for i in range(len(angle)):
            a = angle[i] + self.bias[i]
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            ret.append(a)
        return ret

    def move_joint6_2_joint5(self, pos, ori):
        ret = np.array(
            [
                -ori[0, 2] * self.a6 + pos[0],
                -ori[1, 2] * self.a6 + pos[1],
                -ori[2, 2] * self.a6 + pos[2],
            ]
        )
        return ret

    def j3_ik(self, pos):
        angle = [0.0] * 3
        ret = []

        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            c3 = (
                pos[0] ** 2
                + pos[1] ** 2
                + (pos[2] - self.a1) ** 2
                - self.a3**2
                - self.a4**2
            ) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError("Fail to solve inverse kinematics")

            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3**2)
                angle[2] = np.arctan2(s3, c3)
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                angle[1] = np.arctan2(
                    k1 * (pos[2] - self.a1)
                    - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                    i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                    + k2 * (pos[2] - self.a1),
                )
                js = self.add_bias(angle)
                if np.all(
                    (js > self.arm_joint_range[0, :3])
                    * (js < self.arm_joint_range[1, :3])
                ):
                    ret.append(js)
        return ret


if __name__ == "__main__":
    import os

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    arm_fik = AirbotPlayFIK(
        urdf=os.path.join(
            os.path.dirname(__file__),
            "../../models/urdf/airbot_play_v3_gripper_fixed.urdf",
        )
    )

    # qq = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qq = np.array([-0.0, -0.166, 0.032, 0.0, 1.5708, 2.223 - np.pi / 2.0])

    omi = arm_fik.forwardKin(qq)
    tmat = arm_fik.properFK(qq)
    print(">>> proper fk:")
    print("trans =\n", tmat[:3, 3])
    print("rot   =\n", tmat[:3, :3])

    # result = arm_fik.inverseKin(omi.translation.T, omi.rotation, qq)
    # print(">>> ik res:")
    # print(np.array(result))
