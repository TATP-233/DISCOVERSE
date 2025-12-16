# transport.py
from __future__ import annotations
import asyncio, json
from typing import Any, Dict, Optional

class RpcError(Exception): ...
class TransportClosed(RpcError): ...

class PersistentTransport:
    """
    单连接复用，多请求并发：一次 connect()，后续所有 request() 共用这一根 TCP。
    - request() 自动加 req_id；后台读循环按 req_id 分发应答
    - 支持 auto_reconnect（简单版）
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8888, *, auto_reconnect: bool = True):
        self.host, self.port = host, port
        self.auto_reconnect = auto_reconnect
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._rx_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._closed = False

    async def connect(self):
        if self._reader and not self._reader.at_eof() and self._rx_task and not self._rx_task.done():
            return
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        if self._rx_task:
            self._rx_task.cancel()
            try:
                await self._rx_task
            except Exception:
                pass
        self._rx_task = asyncio.create_task(self._rx_loop())

    async def close(self):
        self._closed = True
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        if self._rx_task:
            self._rx_task.cancel()
            try: await self._rx_task
            except Exception: pass
        self._rx_task = None
        for fut in self._pending.values():
            if not fut.done(): fut.set_exception(TransportClosed("transport closed"))
        self._pending.clear()
        self._reader = None

    async def _ensure_open(self):
        if self._closed:
            raise TransportClosed("transport already closed")
        if not self._reader or not self._writer:
            await self.connect()

    async def _rx_loop(self):
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    raise TransportClosed("server closed connection")
                resp = json.loads(line.decode())
                rid = resp.get("req_id")
                fut = self._pending.pop(rid, None)
                if fut and not fut.done():
                    fut.set_result(resp)
        except Exception as e:
            # 让所有挂起请求失败
            for fut in list(self._pending.values()):
                if not fut.done(): fut.set_exception(e)
            self._pending.clear()
            self._reader = self._writer = None
            if self.auto_reconnect and not self._closed:
                # 下次 request 再重新连接
                pass

    async def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_open()
        loop = asyncio.get_running_loop()
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        async with self._write_lock:
            msg = json.dumps({"req_id": rid, **payload}) + "\n"
            self._writer.write(msg.encode())
            await self._writer.drain()
        return await fut