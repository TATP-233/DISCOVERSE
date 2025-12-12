# manager.py
import asyncio
from typing import Dict, Tuple, Iterable, Optional, Any, List
from discoverse.examples.motor.client.transport import PersistentTransport
from discoverse.examples.motor.client.motor_client import Motor

Key = Tuple[str, int]  # (motor_type, motor_id)

class MotorClientManager:
    """
    多电机共享一条持久连接：增删查、批量并发、批量参数/控制。
    """
    def __init__(self, host="127.0.0.1", port=8888, *, auto_reconnect: bool = True):
        self.tx = PersistentTransport(host, port, auto_reconnect=auto_reconnect)
        self._motors: Dict[Key, Motor] = {}

    async def connect(self):
        await self.tx.connect()

    async def close(self):
        await self.tx.close()

    def add(self, motor_type: str, motor_id: int) -> Motor:
        key: Key = (motor_type, motor_id)
        if key in self._motors:
            return self._motors[key]
        m = Motor.create(motor_type, motor_id, shared_transport=self.tx)
        if m is None:
            raise ValueError(f"unsupported (type,id)=({motor_type},{motor_id})")
        self._motors[key] = m
        return m

    def get(self, motor_type: str, motor_id: int) -> Optional[Motor]:
        return self._motors.get((motor_type, motor_id))

    def all(self) -> Iterable[Motor]:
        return self._motors.values()

    async def _gather_ok(self, coros: List) -> List[bool]:
        rs = await asyncio.gather(*coros, return_exceptions=True)
        return [bool(r) if not isinstance(r, Exception) else False for r in rs]

    # ---- 批量动作 ----
    async def init_all(self, interface: str = "", spin_freq: float = 200.0) -> List[bool]:
        return await self._gather_ok([m.init(interface=interface, spin_freq=spin_freq) for m in self.all()])

    async def uninit_all(self) -> List[bool]:
        return await self._gather_ok([m.uninit() for m in self.all()])

    async def enable_all(self, on: bool = True) -> List[bool]:
        return await self._gather_ok([(m.enable() if on else m.disable()) for m in self.all()])

    async def csp_batch(self, items: list[tuple[str, int, float]]) -> List[bool]:
        coros = []
        for t, i, q in items:
            m = self.get(t, i) or self.add(t, i)
            coros.append(m.csp({"position": float(q)}))
        return await self._gather_ok(coros)

    async def set_param_all(self, name: str, value: Any) -> List[bool]:
        return await self._gather_ok([m.set_param(name, value) for m in self.all()])

async def test_comm_freq(mgr: MotorClientManager, count: int = 1000, concurrent: int = 20) -> float:
    """测试通信频率：返回每秒请求数"""
    import time
    m = next(iter(mgr.all()), None) or mgr.add("EC", 1)
    start = time.perf_counter()
    for i in range(0, count, concurrent):
        await asyncio.gather(*[m.ping() for _ in range(min(concurrent, count - i))])
    elapsed = time.perf_counter() - start
    return count / elapsed
    
async def main():
    mgr = MotorClientManager("127.0.0.1", 8890)
    await mgr.connect()  # 建立一次连接

    # 注册多个电机（共享同一连接）
    for i in range(1, 8):
        mgr.add("EC", i)
    print(" set csp rate:", await mgr.set_param_all("slew_rate", 0.03))

    print("init_all   :", await mgr.init_all(interface="can0", spin_freq=500.0))
    print("enable_all :", await mgr.enable_all(True))

    # 批量 CSP（示例值）
    batch = [("EC", 1, 0.4), ("EC", 2, -0.2), ("EC", 3, 0.3),
             ("EC", 4, 0.3), ("EC", 5,  0.1), ("EC", 6, 0.5), ("EC", 7, 1.0)]
    print("csp_batch  :", await mgr.csp_batch(batch))
    print("disable_all:", await mgr.enable_all(False))
    print("get ping:", await mgr.get("EC",1).ping())

    print("get setZero:", await mgr.get("EC",6).set_zero())
    # print("uninit_all :", await mgr.uninit_all())
    print("all motors  :", list(mgr.all()))

    print("get state EC-3:", await mgr.get("EC",3).state())
    
    print("enable all", await mgr.enable_all(True))
    # 测试通信频率
    freq = await test_comm_freq(mgr, count=1000, concurrent=20)
    print(f"通信频率: {freq:.1f} req/s")
    
    await mgr.close()

if __name__ == "__main__":
    asyncio.run(main())