import os
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation
from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.airbot_play import AirbotPlayFIK


class MMK2FIK:
    TMat_footprint2chest = np.array(
        [
            [1.0, 0.0, 0.0, 0.02371],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.311],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    TMat_endpoint2camera = np.array(
        [
            [0.0, -0.5, 0.86603, -0.105],
            [-1.0, -0.0, 0.0, 0.0],
            [0.0, -0.86603, -0.5, 0.082],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    TMat_chest2lft_base = np.array(
        [
            [0.70711, 0.0, 0.70711, 0.0265],
            [-0.70711, -0.0, 0.70711, 0.095],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    TMat_chest2rgt_base = np.array(
        [
            [0.70711, 0.0, 0.70711, 0.0265],
            [0.70711, -0.0, -0.70711, -0.095],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    action_rot = {
        "pick": {
            "l": np.array(
                [
                    [0.0, 0.7071, 0.7071],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.7071, -0.7071],
                ]
            ),
            "r": np.array(
                [
                    [0.0, -0.7071, 0.7071],
                    [-1.0, 0.0, 0.0],
                    [0.0, -0.7071, -0.7071],
                ]
            ),
        },
        "carry": {
            "l": np.array(
                [
                    [0.0, -0.7071, 0.7071],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.7071, 0.7071],
                ]
            ),
            "r": np.array(
                [
                    [0.0, 0.7071, 0.7071],
                    [-1.0, 0.0, 0.0],
                    [0.0, -0.7071, 0.7071],
                ]
            ),
        },
        "look": {
            "l": np.array(
                [
                    [0.707, -0.707, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.707, 0.707, 0.0],
                ]
            ),
            "r": np.array(
                [
                    [0.707, 0.707, -0.0],
                    [0.0, 0.0, 1.0],
                    [0.707, -0.707, 0.0],
                ]
            ),
        },
    }

    def __init__(self) -> None:
        self.arm_fik = AirbotPlayFIK(
            urdf=os.path.join(
                DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
            )
        )

    def get_3dposition_wrt_arm_base(self, point3d, q):
        """
        point3d : position of the object in the camera frame (z is the depth-axis, x is the horizontal-axis rightward, y is the vertical-axis downward)
        q       : joint angles of the arm
        """
        point3d_homogeneous = np.append(point3d, 1)
        Tmat_armbase2endpoint = self.arm_fik.properFK(q)
        point3d_arm_base = (
            Tmat_armbase2endpoint @ MMK2FIK.TMat_endpoint2camera @ point3d_homogeneous
        )[:3]
        return point3d_arm_base

    def get_3dposition_wrt_footprint(self, point3d, q, slide: float, arm: str):
        """
        point3d : position of the object in the camera frame
        q       : joint angles of the arm
        """
        tmat = MMK2FIK.TMat_footprint2chest.copy()
        tmat[2, 3] -= slide
        if arm == "l":
            Tmat_footprint2armbase = tmat @ MMK2FIK.TMat_chest2lft_base
        elif arm == "r":
            Tmat_footprint2armbase = tmat @ MMK2FIK.TMat_chest2rgt_base
        else:
            raise ValueError("Invalid arm")
        point3d_homogeneous = np.append(point3d, 1)
        Tmat_armbase2endpoint = self.arm_fik.properFK(q)
        point3d_footprint = (
            Tmat_footprint2armbase
            @ Tmat_armbase2endpoint
            @ MMK2FIK.TMat_endpoint2camera
            @ point3d_homogeneous
        )[:3]
        return point3d_footprint

    def get_armjoint_pose_wrt_armbase(
        self,
        point3d,
        action: Union[str, np.ndarray],
        arm: str,
        q_ref=None,
        action_rot=np.eye(3),
    ):
        if type(action) == str:
            rot = MMK2FIK.action_rot[action][arm] @ action_rot
        elif type(action) == np.ndarray:
            rot = action @ action_rot
        jq = self.arm_fik.properIK(point3d, rot, q_ref)
        return jq

    def get_armjoint_pose_wrt_footprint(
        self,
        point3d,
        action: Union[str, np.ndarray],
        arm: str,
        slide: float,
        q_ref=np.zeros(6),
        action_rot=np.eye(3),
    ):
        tmat = MMK2FIK.TMat_footprint2chest.copy()
        tmat[2, 3] -= slide
        if arm == "l":
            Tmat_footprint2armbase = tmat @ MMK2FIK.TMat_chest2lft_base
        elif arm == "r":
            Tmat_footprint2armbase = tmat @ MMK2FIK.TMat_chest2rgt_base
        else:
            raise ValueError("Invalid arm")
        point3d_homogeneous = np.append(point3d, 1)
        point3d_arm_base = (
            np.linalg.inv(Tmat_footprint2armbase) @ point3d_homogeneous
        )[:3]
        try:
            jq = self.get_armjoint_pose_wrt_armbase(
                point3d_arm_base, action, arm, q_ref, action_rot
            )
        except ValueError:
            print(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            print("point3d wrt world :", point3d)
            print("point3d_arm_base  :", point3d_arm_base)
            print("action            :", action)
            print("arm               :", arm)
            print("slide             :", slide)
            print("q_ref             :", q_ref)
            print("action_rot        :\n", action_rot)
            print(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            raise ValueError("Invalid target position")
        return jq


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    arm_sel = "r"
    action = "carry"
    mmk2_func = MMK2FIK()

    pose_wrt_camera = [0, -0.23645, 0]
    # retpose = mmk2_func.get_3dposition_wrt_arm_base(pose_wrt_camera, np.zeros(6))
    # print(retpose)
    # jq = mmk2_func.get_armjoint_pose_wrt_armbase(retpose, action, arm_sel, np.zeros(6))
    # print(np.array(jq))

    # omi = mmk2_func.arm_fik.properFK(jq)
    # print(omi)
    # print(mmk2_func.arm_fik.arm_rot_mat)

    # pose_wrt_camera = [0.41024, -0.19778,  1.27201]
    # retpose = mmk2_func.get_3dposition_wrt_footprint(pose_wrt_camera, np.zeros(6), 0, arm_sel)
    # print(retpose)

    retpose = np.array([0.5, -0.2, 1.15])
    art = Rotation.from_euler("zyx", [0, -1.1, 0]).as_matrix()
    jq = mmk2_func.get_armjoint_pose_wrt_footprint(
        retpose, action, arm_sel, 0, np.zeros(6), art
    )
    print(np.array(jq))
    print(np.array2string(np.array(jq), separator=", ", precision=5))

    # retpose = mmk2_func.get_3dposition_wrt_footprint([0, 0, 0], np.zeros(6), 0, "l")
    # print(retpose)
    # retpose = mmk2_func.get_3dposition_wrt_footprint([0, 0, 0], np.zeros(6), 0, "r")
    # print(retpose)
