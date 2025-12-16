from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .transport import PersistentTransport
from .motor_client import Motor, MotorState, MotorType, EEFType


MotorSpec = Union[Tuple[str, int], Motor]


@dataclass
class PlayState:
    is_valid: bool
    pos: List[float]
    vel: List[float]
    eff: List[float]

    @property
    def count(self) -> int:
        return len(self.pos)


class Play:
    """
    机械臂（6关节 + 夹爪）门面，基于 Motor API 组合实现。
    """

    def __init__(self, transport: PersistentTransport, joints: List[Motor], eef: Motor):
        if len(joints) != 6:
            raise ValueError("Play 期望正好 6 个关节电机")
        self.tx = transport
        self._joints = joints
        self._eef = eef

    # ---- 工厂 ----
    @staticmethod
    def create(
        m1: MotorType,
        m2: MotorType,
        m3: MotorType,
        m4: MotorType,
        m5: MotorType,
        m6: MotorType,
        eef_type: EEFType,
        eef_motor_type: MotorType,
        *,
        host: str = "127.0.0.1",
        port: int = 8890,
        auto_reconnect: bool = True,
    ) -> "Play":
        """
        创建 Play 实例。
        
        参数:
            m1-m6: 6个关节电机的类型（MotorType 枚举）
            eef_type: 末端执行器类型（EEFType 枚举）
            eef_motor_type: 末端执行器电机类型（MotorType 枚举）
        
        自动 ID 分配:
            - 关节 1-6 对应 motor_id 1-6
            - 末端执行器对应 motor_id 7
        """
        joint_types = [m1, m2, m3, m4, m5, m6]
        shared_transport = PersistentTransport(host, port, auto_reconnect=auto_reconnect)
        
        # 创建 6 个关节电机，id 从 1 到 6
        joints = []
        for idx, motor_type in enumerate(joint_types, start=1):
            motor = Motor.create(
                motor_type,
                idx,
                host=host,
                port=port,
                auto_reconnect=auto_reconnect,
                shared_transport=shared_transport,
            )
            if motor is None:
                raise ValueError(f"unsupported motor type {motor_type} with id {idx}")
            joints.append(motor)
        
        # 创建末端执行器
        eef = Motor.create(
            eef_motor_type,
            7,
            host=host,
            port=port,
            auto_reconnect=auto_reconnect,
            shared_transport=shared_transport,
        )
        if eef is None:
            raise ValueError(f"unsupported end-effector motor type {eef_motor_type} with id 7")
        
        return Play(shared_transport, joints, eef)

    # ---- 生命周期 ----
    async def connect(self):
        await self.tx.connect()

    async def close(self):
        await self.tx.close()

    async def init(self, io_context=None, interface: str = "", spin_freq: float = 200.0) -> bool:
        coros = [m.init(interface=interface, spin_freq=spin_freq) for m in self._motors()]
        return await _all_ok(coros)

    async def uninit(self) -> bool:
        coros = [m.uninit() for m in self._motors()]
        return await _all_ok(coros)

    async def ping(self) -> bool:
        motors = list(self._motors())
        if not motors:
            return False
        return await _all_ok([m.ping() for m in motors])

    # ---- 使能 / 清错 ----
    async def enable(self) -> bool:
        return await _all_ok([m.enable() for m in self._motors()])

    async def disable(self) -> bool:
        """禁用所有电机（包括所有关节和末端执行器）"""
        return await _all_ok([m.disable() for m in self._motors()])

    async def set_zero(self) -> bool:
        return await _all_ok([m.set_zero() for m in self._motors()])

    async def reset_error(self) -> bool:
        return await _all_ok([m.reset_error() for m in self._motors()])

    # ---- 控制 ----
    async def csv(self, velocities: Sequence[float]) -> bool:
        targets = _normalize_targets(velocities, self.joint_count, self.total_count)
        coros = [m.csv({"velocity": float(v)}) for m, v in zip(self._motors(), targets)]
        return await _all_ok(coros)

    async def csp(self, positions: Sequence[float]) -> bool:
        targets = _normalize_targets(positions, self.joint_count, self.total_count)
        coros = [m.csp({"position": float(q)}) for m, q in zip(self._motors(), targets)]
        return await _all_ok(coros)

    async def pvt(
        self,
        pos: Sequence[float],
        max_vel: Optional[Sequence[float]] = None,
        max_eff: Optional[Sequence[float]] = None,  # 占位：协议暂未使用
    ) -> bool:
        positions = _normalize_targets(pos, self.joint_count, self.total_count)
        velocities = _normalize_optional(max_vel, len(positions), default=0.0)
        coros = [
            m.pvt({"q_ref": float(q), "dq_ref": float(dq)})
            for m, q, dq in zip(self._motors(), positions, velocities)
        ]
        return await _all_ok(coros)

    async def mit(self, commands: Sequence[Dict[str, Any]]) -> bool:
        if len(commands) != self.total_count:
            raise ValueError(f"MIT 命令需要 {self.total_count} 份参数")
        coros = []
        for motor, cmd in zip(self._motors(), commands):
            payload = {
                "q_ref": float(cmd.get("q_ref", 0.0)),
                "dq_ref": float(cmd.get("dq_ref", 0.0)),
                "kp": float(cmd.get("kp", 0.0)),
                "kd": float(cmd.get("kd", 0.0)),
                "tau_ff": float(cmd.get("tau_ff", 0.0)),
            }
            coros.append(motor.mit(payload))
        return await _all_ok(coros)

    # ---- 参数 ----
    async def get_param(self, name: str) -> List[Any]:
        return await asyncio.gather(*(m.get_param(name) for m in self._motors()))

    async def set_param(self, name: str, value: Any) -> bool:
        return await _all_ok([m.set_param(name, value) for m in self._motors()])

    async def persist_param(self, name: str, value: Any) -> bool:
        return await _all_ok([m.persist_param(name, value) for m in self._motors()])

    async def params(self) -> List[Dict[str, Any]]:
        return await asyncio.gather(*(m.params() for m in self._motors()))

    # ---- 状态 ----
    async def state(self) -> PlayState:
        motors = list(self._motors())
        total = len(motors)

        if total == 0:
            return PlayState(
                is_valid=False,
                pos=[],
                vel=[],
                eff=[],
            )

        states = await asyncio.gather(*(m.state() for m in motors), return_exceptions=True)
        
        positions = []
        velocities = []
        forces = []
        all_valid = True

        for result in states:
            # 统一转换为 MotorState
            if isinstance(result, MotorState):
                motor_state = result
            elif isinstance(result, Exception):
                motor_state = MotorState(is_valid=False)
            else:
                motor_state = MotorState(is_valid=False)
            
            if not motor_state.is_valid:
                all_valid = False
            
            positions.append(float(motor_state.position))
            velocities.append(float(motor_state.velocity))
            forces.append(float(motor_state.torque))

        return PlayState(
            is_valid=all_valid,
            pos=positions,
            vel=velocities,
            eff=forces,
        )

    # ---- 属性 ----
    @property
    def joint_count(self) -> int:
        return len(self._joints)

    @property
    def has_eef(self) -> bool:
        return True

    @property
    def total_count(self) -> int:
        return self.joint_count + 1

    def _motors(self) -> Iterable[Motor]:
        return [*self._joints, self._eef]


class PlayWithEEF(Play):
    """
    带末端执行器的机械臂（6关节 + 末端执行器）门面。
    这是 Play 的别名，用于保持向后兼容性。
    """

    @staticmethod
    def create(
        m1: MotorType,
        m2: MotorType,
        m3: MotorType,
        m4: MotorType,
        m5: MotorType,
        m6: MotorType,
        eef_type: EEFType,
        eef_motor_type: MotorType,
        *,
        host: str = "127.0.0.1",
        port: int = 8890,
        auto_reconnect: bool = True,
    ) -> "PlayWithEEF":
        """
        创建带末端执行器的 Play 实例。
        
        参数:
            m1-m6: 6个关节电机的类型（MotorType 枚举）
            eef_type: 末端执行器类型（EEFType 枚举）
            eef_motor_type: 末端执行器电机类型（MotorType 枚举）
        
        自动 ID 分配:
            - 关节 1-6 对应 motor_id 1-6
            - 末端执行器对应 motor_id 7
        """
        play = Play.create(
            m1, m2, m3, m4, m5, m6,
            eef_type,
            eef_motor_type,
            host=host,
            port=port,
            auto_reconnect=auto_reconnect,
        )
        return PlayWithEEF(play.tx, play._joints, play._eef)


def _shared_transport_from_specs(specs: Sequence[MotorSpec]) -> Optional[PersistentTransport]:
    for spec in specs:
        if isinstance(spec, Motor):
            return spec.tx
    return None


def _ensure_motor(
    spec: MotorSpec,
    transport: PersistentTransport,
    host: str,
    port: int,
    auto_reconnect: bool,
) -> Motor:
    if isinstance(spec, Motor):
        if spec.tx is not transport:
            raise ValueError("传入的 Motor 不共享同一个 transport")
        return spec
    if not isinstance(spec, tuple) or len(spec) != 2:
        raise ValueError("Motor 规格需为 ('TYPE', id) 或已有 Motor 实例")
    motor_type, motor_id = spec
    motor = Motor.create(
        motor_type,
        int(motor_id),
        host=host,
        port=port,
        auto_reconnect=auto_reconnect,
        shared_transport=transport,
    )
    if motor is None:
        raise ValueError(f"unsupported motor spec ({motor_type}, {motor_id})")
    return motor


def _normalize_targets(values: Sequence[float], joint_count: int, total_count: int) -> List[float]:
    data = list(values)
    if len(data) not in (joint_count, total_count):
        raise ValueError(f"目标数量必须是 {joint_count} 或 {total_count}")
    if len(data) == joint_count and total_count > joint_count:
        data.append(0.0)
    return [float(v) for v in data]


def _normalize_optional(values: Optional[Sequence[float]], length: int, default: float) -> List[float]:
    if values is None:
        return [default] * length
    data = list(values)
    if len(data) not in (1, length):
        raise ValueError(f"可选参数长度必须为 1 或 {length}")
    if len(data) == 1:
        data = data * length
    return [float(v) for v in data]


async def _all_ok(coroutines: Iterable[Awaitable[Any]]) -> bool:
    coro_list = list(coroutines)
    if not coro_list:
        return True
    results = await asyncio.gather(*coro_list, return_exceptions=True)
    ok = True
    for r in results:
        if isinstance(r, Exception):
            ok = False
        elif not bool(r):
            ok = False
    return ok


__all__ = ["Play", "PlayWithEEF", "PlayState", "MotorType", "EEFType"]
