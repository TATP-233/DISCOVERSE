from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from discoverse.examples.motor.client.transport import PersistentTransport

class MotorType(str, Enum):
    OD  = "OD"
    DM  = "DM"
    ODM = "ODM"
    EC  = "EC"


class EEFType(str, Enum):
    """末端执行器类型枚举"""
    G2 = "G2"  # G2 夹爪

# 合法 (type,id) 组合
VALID_IDS = {
    MotorType.OD:  {1, 2, 3, 7},
    MotorType.DM:  {4, 5, 6, 7},
    MotorType.ODM: {4, 5, 6, 7},
    MotorType.EC:  {1, 2, 3, 4, 5, 6, 7},
}

def normalize_motor_type(t) -> MotorType | None:
    if isinstance(t, MotorType):
        return t
    if isinstance(t, str):
        s = t.strip().upper()
        try:
            return MotorType[s]      # "EC" -> MotorType.EC
        except KeyError:
            try:
                return MotorType(s)  # 枚举值字符串
            except Exception:
                return None
    return None

# motor_client.py
class MotorError(Exception): ...

@dataclass
class MotorState:
    is_valid: bool = False
    joint_id: int = 0
    pos: float = 0.0
    vel: float = 0.0
    eff: float = 0.0
    motor_temp: int = 0
    mos_temp: int = 0
    enabled: bool = True
    mode: Optional[str] = None
    temperature: float = 0.0
    voltage: float = 0.0
    current: float = 0.0
    error_code: int = 0
    error_message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> float:
        return self.pos

    @property
    def velocity(self) -> float:
        return self.vel

    @property
    def torque(self) -> float:
        return self.eff

class Motor:
    """
    与 C++/pybind Motor API 对齐的门面（纯客户端）。依赖持久连接 transport。
    """
    def __init__(self, transport: PersistentTransport, motor_type: MotorType, motor_id: int):
        if motor_id not in VALID_IDS.get(motor_type, set()):
            raise MotorError(f"unsupported (type,id)=({motor_type},{motor_id})")
        self.tx = transport
        self.motor_type = motor_type
        self.motor_id   = int(motor_id)
        self._initialized = False

    # ---- 静态工厂（非法返回 None，仿 C++ .create）----
    @staticmethod
    def create(motor_type: Union[str, MotorType], motor_id: int,
               host: str = "127.0.0.1", port: int = 8888,
               *, auto_reconnect: bool = True,
               shared_transport: Optional[PersistentTransport] = None) -> Optional["Motor"]:
        mt = normalize_motor_type(motor_type)
        if mt is None: return None
        if motor_id not in VALID_IDS.get(mt, set()): return None
        tx = shared_transport or PersistentTransport(host, port, auto_reconnect=auto_reconnect)
        return Motor(tx, mt, motor_id)

    # ---- 生命周期 / 心跳 ----
    async def init(self, io_context=None, interface: str = "", spin_freq: float = 200.0) -> bool:
        resp = await self.tx.request({
            "type": "Init",
            "motor_id": self.motor_id,
            "interface": interface,
            "spin_freq": spin_freq,
        })
        ok = bool(resp.get("ok", False))
        if ok:
            self._initialized = True
        return ok

    async def uninit(self) -> bool:
        if not self._initialized:
            return True
        resp = await self.tx.request({"type": "Uninit", "motor_id": self.motor_id})
        ok = bool(resp.get("ok", False))
        if ok:
            self._initialized = False
            # await self.tx.close()
        return ok

    async def ping(self) -> bool:
        resp = await self._request({"type": "Ping"})
        # resp = await self.tx.request({"type": "Ping"})
        return bool(resp.get("ok", False))

    # ---- 使能 / 零点 / 清错 ----
    async def enable(self) -> bool:
        resp = await self._request({"type": "Enable", "motor_id": self.motor_id})
        return bool(resp.get("ok", False))

    async def disable(self) -> bool:
        resp = await self._request({"type": "Disable", "motor_id": self.motor_id})
        return bool(resp.get("ok", False))

    async def set_zero(self) -> bool:
        resp = await self._request({"type": "SetZero", "motor_id": self.motor_id})
        return bool(resp.get("ok", False))

    async def reset_error(self) -> bool:
        resp = await self._request({"type": "ResetError", "motor_id": self.motor_id})
        return bool(resp.get("ok", False))

    # ---- 控制（CSP/CSV/MIT/PVT）----
    async def csp(self, cmd: Dict[str, Any]) -> bool:
        resp = await self._request({"type": "CSP", "motor_id": self.motor_id, **cmd})
        return bool(resp.get("ok", False))

    async def csv(self, cmd: Dict[str, Any]) -> bool:
        resp = await self._request({"type": "CSV", "motor_id": self.motor_id, **cmd})
        return bool(resp.get("ok", False))

    async def mit(self, cmd: Dict[str, Any]) -> bool:
        resp = await self._request({"type": "MIT", "motor_id": self.motor_id, **cmd})
        return bool(resp.get("ok", False))

    async def pvt(self, cmd: Dict[str, Any]) -> bool:
        resp = await self._request({"type": "PVT", "motor_id": self.motor_id, **cmd})
        return bool(resp.get("ok", False))

    # ---- 参数 ----
    async def get_param(self, name: str) -> Any:
        resp = await self.tx.request({"type": "GetParam", "motor_id": self.motor_id, "name": name})
        return (resp.get("data") or {}).get("value")

    async def set_param(self, name: str, value: Any) -> bool:
        resp = await self.tx.request({"type": "SetParam", "motor_id": self.motor_id, "name": name, "value": value})
        return bool(resp.get("ok", False))

    async def persist_param(self, name: str, value: Any) -> bool:
        resp = await self.tx.request({"type": "PersistParam", "motor_id": self.motor_id, "name": name, "value": value})
        return bool(resp.get("ok", False))

    # ---- 状态 ----
    async def update(self, result: Dict[str, Any]) -> bool:
        resp = await self._request({"type": "Update", "motor_id": self.motor_id, "result": result})
        return bool(resp.get("ok", False))

    async def state(self) -> MotorState:
        resp = await self._request({"type": "GetState", "motor_id": self.motor_id})
        ok = bool(resp.get("ok", False))
        data = (resp.get("data") or {}) if ok else {}
        mode_val = data.get("mode")
        return MotorState(
            is_valid=ok,
            joint_id=int(data.get("joint_id", self.motor_id)),
            pos=float(
                data.get(
                    "pos",
                    data.get("position", 0.0),
                )
            ),
            vel=float(
                data.get(
                    "vel",
                    data.get("velocity", 0.0),
                )
            ),
            eff=float(
                data.get(
                    "eff",
                    data.get("torque", 0.0),
                )
            ),
            motor_temp=int(data.get("motor_temp", 0) or 0),
            mos_temp=int(data.get("mos_temp", 0) or 0),
            enabled=bool(data.get("enabled", data.get("is_enabled", True))),
            mode=str(mode_val) if mode_val is not None else None,
            temperature=float(data.get("temperature", 0.0) or 0.0),
            voltage=float(data.get("voltage", 0.0) or 0.0),
            current=float(data.get("current", 0.0) or 0.0),
            error_code=int(data.get("error_code", 0) or 0),
            error_message=str(data.get("error_message", "") or ""),
            raw=data if ok else resp,
        )

    async def params(self) -> Dict[str, Any]:
        resp = await self._request({"type": "GetParams", "motor_id": self.motor_id})
        return resp.get("data", {}) if resp.get("ok", False) else {}

    async def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initialized:
            raise MotorError("motor not initialized; call init() first")
        return await self.tx.request(payload)

# ---- 模块级工厂（与 C++ create 语义一致）----
# def create_motor_runtime(motor_type: Union[str, MotorType], motor_id: int,
#                          host: str = "127.0.0.1", port: int = 8888,
#                          *, auto_reconnect: bool = True,
#                          shared_transport: Optional[PersistentTransport] = None) -> Optional[Motor]:
#     mt = normalize_motor_type(motor_type)
#     if mt is None: return None
#     if motor_id not in VALID_IDS.get(mt, set()): return None
#     tx = shared_transport or PersistentTransport(host, port, auto_reconnect=auto_reconnect)
#     return Motor(tx, mt, motor_id)