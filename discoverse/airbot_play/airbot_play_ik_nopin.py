import numpy as np


class AirbotPlayIK_nopin:
    bias = np.array([0.0, -2.7549, 2.7549, 1.5708, 0.0, 0.0])
    a1 = 0.1172
    a3 = 0.27009
    a4 = 0.29015
    a6 = 0.23645
    arm_joint_range = np.array(
        [
            [-3.14, -2.96, -0.087, -2.96, -1.74, -3.14],
            [2.09, 0.17, 3.14, 2.96, 1.74, 3.14],
        ]
    )
    joint_range_scale = arm_joint_range[1] - arm_joint_range[0]

    arm_rot_mat = np.array([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def __init__(self, urdf) -> None:
        pass

    def properIK(self, pos, ori, ref_q=None):
        return self.inverseKin(pos, ori @ self.arm_rot_mat, ref_q)

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
    arm_ik = AirbotPlayIK_nopin()

    trans = np.array([0.276, -0.0, 0.219])
    rot = np.array([[1.0, -0.0, -0.0], [0.0, 1.0, -0.0], [0.0, 0.0, 1.0]])

    qs = arm_ik.properIK(trans, rot)

    for q in qs:
        print(q)
