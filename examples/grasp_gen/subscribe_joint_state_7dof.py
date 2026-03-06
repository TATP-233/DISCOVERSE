# subscribe_joint_state_7dof.py
"""
7DOF Play G3 专用：订阅 robot/state/joint 与 gripper/sim_two_finger/state，转换为 8 维 action。

gripper/sim_two_finger/state position 语义：
  - position=0       → 夹爪闭合（关闭）
  - position=0.0722  → 夹爪张开（开启）

Payload 格式支持：
  - 14 float32：7q + 7v（仅臂），gripper 来自 gripper topic
  - 18 float32：9q + 9v（7 臂 + 2 夹爪指），gripper 从 q[7],q[8] 推导 width

Output action (8,):
  - action[:7] = arm q (rad)
  - action[7]  = gripper width [0, 0.0722]，consumer 需 width/GRIPPER_WIDTH_MAX 转为 sim ctrl [0,1]
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import zenoh

import generated.gripper_2f_pb2 as gripper_pb


ARM_DOF = 7
# gripper/sim_two_finger/state position: 0=闭合, 0.0722=张开
GRIPPER_WIDTH_MIN = 0.0
GRIPPER_WIDTH_MAX = 0.0722
# robot/state/joint 18-float 模式：endleft[-0.04,0] endright[0,0.04]，span=0.08 对应 width=0.0722
GRIPPER_JOINT_SPAN = 0.08


@dataclass
class JointState7dof:
    q: np.ndarray  # (7,) 或含 gripper 时 (9,) 全量
    v: np.ndarray  # (7,)
    q_gripper: Optional[tuple]  # (q7, q8) 当 payload 18 时，否则 None


def decode_robot_state_joint_7dof(payload: bytes) -> JointState7dof:
    """
    Decode payload from topic 'robot/state/joint'.

    支持:
      - 14 float32: [q(7), v(7)]
      - 18 float32: [q(9), v(9)]，9 = 7 臂 + 2 夹爪指
    """
    vec = np.frombuffer(payload, dtype=np.float32)
    if vec.size % 2 != 0:
        raise ValueError(f"Payload must have even float32 count, got {vec.size}")
    dof = vec.size // 2
    q_all = vec[:dof].copy()
    v_all = vec[dof:].copy()
    if dof >= 9:
        # 18 floats: 7 arm + 2 gripper
        q = q_all[:7].copy()
        v = v_all[:7].copy()
        q_gripper = (float(q_all[7]), float(q_all[8]))
    elif dof >= 7:
        q = q_all[:7].copy()
        v = v_all[:7].copy()
        q_gripper = None
    else:
        raise ValueError(f"Payload dof={dof} < 7, need at least 7 arm joints")
    return JointState7dof(q=q, v=v, q_gripper=q_gripper)


class ArmGripperSubscriber7dof:
    """
    7DOF Play G3 专用订阅器。

    Output action (8,) = [q0..q6, gripper_width]，
    gripper_width 来自 gripper/sim_two_finger/state：0=闭合，0.0722=张开。
    Consumer 需 width/0.0722 转为 ctrl [0,1]。
    """

    def __init__(
        self,
        arm_topic: str = "robot/state/joint",
        gripper_topic: str = "gripper/sim_two_finger/state",
        gripper_ctrl_min: float = GRIPPER_WIDTH_MIN,
        gripper_ctrl_max: float = GRIPPER_WIDTH_MAX,
        invert_gripper: bool = False,
        fallback_arm_q: Optional[np.ndarray] = None,
    ):
        self.arm_topic = arm_topic
        self.gripper_topic = gripper_topic
        self.grip_min = float(gripper_ctrl_min)
        self.grip_max = float(gripper_ctrl_max)
        self.invert_gripper = bool(invert_gripper)
        self._fallback_arm_q = fallback_arm_q.copy() if fallback_arm_q is not None and len(fallback_arm_q) >= 7 else None

        self._lock = threading.Lock()
        self._latest_arm: Optional[JointState7dof] = None
        self._latest_grip_width: float = 0.0
        self._gripper_topic_received: bool = False

        self._session = zenoh.open(zenoh.Config())
        self._sub_arm = self._session.declare_subscriber(
            self.arm_topic, self._on_arm
        )
        self._sub_grip = self._session.declare_subscriber(
            self.gripper_topic, self._on_gripper
        )

    def _on_arm(self, sample):
        try:
            payload = bytes(sample.payload)
            js = decode_robot_state_joint_7dof(payload)
            with self._lock:
                self._latest_arm = js
        except Exception as e:
            print(f"[ArmGripperSubscriber7dof] arm decode error: {e}")

    def _on_gripper(self, sample):
        """gripper/sim_two_finger/state: position 0=闭合, 0.0722=张开，裁剪到 [grip_min, grip_max]"""
        try:
            payload = bytes(sample.payload)
            msg = gripper_pb.Gripper2FState()
            msg.ParseFromString(payload)
            width = float(msg.position)
            width = float(np.clip(width, self.grip_min, self.grip_max))
            if self.invert_gripper:
                width = self.grip_max - width
            with self._lock:
                self._latest_grip_width = width
                self._gripper_topic_received = True
        except Exception as e:
            print(f"[ArmGripperSubscriber7dof] gripper decode error: {e}")

    def get_latest_action(self) -> Optional[np.ndarray]:
        """
        Return action (8,) = [q0..q6, gripper_ctrl]。
        gripper_ctrl 为 sim actuator 范围 [0, 1]：0=闭合，1=张开。
        width [0, GRIPPER_WIDTH_MAX] 在内部映射为 ctrl [0, 1]。
        """
        with self._lock:
            if self._latest_arm is None:
                if self._fallback_arm_q is None:
                    return None
                q = self._fallback_arm_q[:7].astype(np.float32)
                grip_width = self._latest_grip_width
            else:
                q = self._latest_arm.q
                # 已收到 gripper topic 时优先用它；否则用 18-float 中的 q_gripper（无 gripper topic 时）
                if self._gripper_topic_received:
                    grip_width = self._latest_grip_width
                elif self._latest_arm.q_gripper is not None:
                    q7, q8 = self._latest_arm.q_gripper
                    span = abs(q8 - q7)
                    grip_width = np.clip(
                        span * GRIPPER_WIDTH_MAX / GRIPPER_JOINT_SPAN,
                        GRIPPER_WIDTH_MIN, GRIPPER_WIDTH_MAX,
                    )
                    if self.invert_gripper:
                        grip_width = GRIPPER_WIDTH_MAX - grip_width
                else:
                    grip_width = self._latest_grip_width
            # width [0, GRIPPER_WIDTH_MAX] -> ctrl [0, 1]，供 sim actuator 直接使用
            grip_ctrl = np.clip(grip_width / GRIPPER_WIDTH_MAX, 0.0, 1.0)
            action = np.zeros(8, dtype=np.float32)
            action[:7] = q.astype(np.float32)
            # print("grip_ctrl",grip_ctrl)
            action[7] = np.float32(grip_ctrl)
            return action

    def get_latest_raw_arm(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if self._latest_arm is None:
                return None
            return self._latest_arm.q.copy(), self._latest_arm.v.copy()

    def get_latest_gripper_width(self) -> float:
        with self._lock:
            return float(self._latest_grip_width)

    def close(self):
        try:
            self._sub_arm.undeclare()
        except Exception:
            pass
        try:
            self._sub_grip.undeclare()
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass
