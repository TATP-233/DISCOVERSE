#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import argparse
import json
import sys
import time
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple

from airbot_play_shim import AirbotPlayShim, AirbotPlayCfg, ControlMode

# ------------------------- Actor：串行化访问 airbot env -------------------------
@dataclass
class ActorRequest:
    cmd: str
    args: Dict[str, Any]
    fut: "asyncio.Future"

class AirbotActor(threading.Thread):
    def __init__(self, headless: bool, ctrl_hz: float = 200.0):
        super().__init__(daemon=True)
        self._headless = headless
        self._ctrl_dt = 1.0 / float(ctrl_hz)
        self.queue: "queue.Queue[ActorRequest]" = queue.Queue(maxsize=1024)
        self._running = threading.Event(); self._running.set()

        self.env = None
        self.last_obs: Optional[Any] = None
        self.last_info: Dict[str, Any] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def _build_handlers(self):
        env = self.env
        handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

        def _require(k: str, d: Dict[str, Any]):
            if k not in d:
                raise ValueError(f"missing field '{k}'")

        # --- 兼容不同 shim 的 enabled/mode 读法 ---
        def _get_enabled_for_joint(j: int) -> bool:
            """统一通过 1-based 的 get_motor 取 MotorShim，再解析 enabled。"""
            try:
                m = env.get_motor(int(j))  # 1-based
                # 优先以运行时快照为准
                if hasattr(m, "_params_now"):
                    try:
                        pn = m._params_now()
                        if hasattr(pn, "enabled"):
                            return bool(pn.enabled)
                    except Exception:
                        pass
                # 其次从配置上读（不同实现里字段名可能不同）
                for attr in ("param", "params", "cfg", "config"):
                    p = getattr(m, attr, None)
                    if p is not None and hasattr(p, "enabled"):
                        return bool(getattr(p, "enabled"))
                return True
            except Exception:
                return True  # 保守：任何异常都当作已启用，避免影响主流程


        def _get_mode_for_joint(j: int) -> Optional[str]:
            """统一通过 1-based 的 get_motor 取 MotorShim，再解析 mode。"""
            try:
                m = env.get_motor(int(j))  # 1-based
                md = getattr(m, "mode", None)
                if md is None:
                    return None
                return str(getattr(md, "name", md))
            except Exception:
                return None
        # --- 通过 shim 暴露的接口 ---
        def _ensure_mode(j: int, mode: ControlMode):
            """确保关节处于指定模式，如果不是则自动切换"""
            if env.get_motor(j).mode != mode:
                env.set_mode_joint(j, mode)

        def h_csp(a: Dict[str, Any]):
            _require("joint", a); _require("position", a)
            j = int(a["joint"])
            _ensure_mode(j, ControlMode.CSP)
            env.csp_joint(j, float(a["position"])); return {"ok": True}

        def h_csv(a: Dict[str, Any]):
            _require("joint", a); _require("velocity", a)
            if hasattr(env, "csv_joint"):
                j = int(a["joint"])
                _ensure_mode(j, ControlMode.CSV)
                env.csv_joint(j, float(a["velocity"])); return {"ok": True}
            raise NotImplementedError("Shim 未实现 csv_joint")

        def h_mit(a: Dict[str, Any]):
            for k in ("joint","q_ref","dq_ref","kp","kd"): _require(k, a)
            tau_ff = float(a.get("tau_ff", 0.0))
            if hasattr(env, "mit_joint"):
                j = int(a["joint"])
                _ensure_mode(j, ControlMode.MIT)
                env.mit_joint(j, float(a["q_ref"]), float(a["dq_ref"]),
                              float(a["kp"]), float(a["kd"]), tau_ff); return {"ok": True}
            raise NotImplementedError("Shim 未实现 mit_joint")

        def h_pvt(a: Dict[str, Any]):
            for k in ("joint","q_ref","dq_ref"): _require(k, a)
            if hasattr(env, "pvt_joint"):
                j = int(a["joint"])
                _ensure_mode(j, ControlMode.PVT)
                env.pvt_joint(j, float(a["q_ref"]), float(a["dq_ref"])); return {"ok": True}
            raise NotImplementedError("Shim 未实现 pvt_joint")

        def h_set_mode(a: Dict[str, Any]):
            _require("joint", a); _require("mode", a)
            if hasattr(env, "set_mode_joint"):
                env.set_mode_joint(int(a["joint"]), a["mode"]); return {"ok": True}
            raise NotImplementedError("Shim 未实现 set_mode_joint")

        def h_enable(a: Dict[str, Any]):
            _require("joint", a); _require("enabled", a)
            env.enable_joint(int(a["joint"]), bool(a["enabled"])); return {"ok": True}

        def h_set_slew(a: Dict[str, Any]):
            _require("joint", a); _require("rate", a)
            env.set_slew_rate_joint(int(a["joint"]), float(a["rate"])); return {"ok": True}

        def h_set_zero(a: Dict[str, Any]):
            _require("joint", a); env.set_zero_joint(int(a["joint"])); return {"ok": True}

        def h_get_state(a: Dict[str, Any]):
            joint = a.get("joint")
            if joint is not None:
                j = int(joint)
                st = env.get_joint_state(j)
                pos = st.get("pos", st.get("position", 0.0))
                vel = st.get("vel", st.get("velocity", 0.0))
                return {
                    "position": pos,
                    "velocity": vel,
                    "torque": 0.0, "temperature": 0.0, "voltage": 0.0, "current": 0.0,
                    "enabled": _get_enabled_for_joint(j), "mode": _get_mode_for_joint(j),
                    "error_code": 0, "error_message": "",
                }
            out: Dict[str, Any] = {"sim_time": getattr(env, "sim_time", None)}
            if hasattr(env, "get_joint_states"):
                out["joints"] = env.get_joint_states()
            elif self.last_obs is not None:
                out["obs"] = self.last_obs
            if self.last_info: out["info"] = self.last_info
            return out

        def h_reset(a: Dict[str, Any]):
            self.last_obs = env.reset(); return {"ok": True}

        def h_quit(a: Dict[str, Any]):
            self._running.clear(); return {"ok": True}

        def h_batch(a: Dict[str, Any]):
            items = a.get("items", []); results = []
            for it in items:
                if not isinstance(it, dict): results.append({"error":"bad item"}); continue
                m, p = normalize_message(it)
                c, aa = MethodRouter.to_actor_cmd(m, p)
                fn = handlers.get(c)
                if fn is None: results.append({"error": f"unknown cmd: {c}"}); continue
                try: results.append(fn(aa))
                except Exception as e: results.append({"error": str(e)})
            return {"ok": True, "results": results}

        # ---- 将 Init/Uninit 语义化为设置/取消网络通信周期（复用 actor 循环节拍）----
        def h_set_hz(a: Dict[str, Any]):
            rate = float(a.get("spin_freq", 200.0))
            if rate <= 0:
                raise ValueError("spin_freq must be > 0")
            self._ctrl_dt = 1.0 / rate
            return {"ok": True, "ctrl_hz": rate}

        def h_unset_hz(a: Dict[str, Any]):
            self._ctrl_dt = 1.0 / 200.0
            return {"ok": True, "ctrl_hz": 200.0}

        handlers.update({
            "csp": h_csp, "csv": h_csv, "mit": h_mit, "pvt": h_pvt,         # 控制
            "set_mode": h_set_mode, "enable": h_enable,
            "set_slew": h_set_slew, "set_zero": h_set_zero, "get_state": h_get_state,
            "reset": h_reset, "quit": h_quit, "batch": h_batch,
            "set_hz": h_set_hz, "unset_hz": h_unset_hz,
            "ping": lambda a: {"pong": True, "t": time.time()},
        })
        self._handlers = handlers

    def run(self):
        cfg = AirbotPlayCfg()
        cfg.headless = bool(self._headless)
        self.env = AirbotPlayShim(cfg)   # 直接实例化
        self._build_handlers()
        try:
            self.last_obs = self.env.reset()
        except Exception:
            self.env.reset(); self.last_obs = None

        last_step = time.perf_counter()
        while self._running.is_set():
            # 拉取请求
            for _ in range(64):
                try:
                    req: ActorRequest = self.queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    handler = self._handlers.get(req.cmd)
                    if handler is None: raise ValueError(f"unknown cmd: {req.cmd}")
                    res = handler(req.args or {})
                    payload = res if isinstance(res, dict) and "ok" in res else {"ok": True, "data": res}
                    loop = req.fut.get_loop()
                    loop.call_soon_threadsafe(req.fut.set_result, payload)
                except Exception as e:
                    req.fut.set_result({"ok": False, "error": str(e)})

            # 推进仿真
            try:
                step_ret = self.env.step(None)
                if isinstance(step_ret, tuple):
                    if len(step_ret) >= 1: self.last_obs = step_ret[0]
                    if len(step_ret) >= 5 and isinstance(step_ret[4], dict): self.last_info = step_ret[4]
            except Exception as e:
                print(f"[actor] step error: {e}", file=sys.stderr)

            # 控制频率
            now = time.perf_counter()
            dt = now - last_step
            if dt < self._ctrl_dt:
                time.sleep(self._ctrl_dt - dt)
            last_step = now

        try:
            if hasattr(self.env, "close"): self.env.close()
        except Exception:
            pass
        print("[actor] stopped")

    def submit(self, cmd: str, args: Optional[Dict[str, Any]], fut: "asyncio.Future"):
        try:
            self.queue.put_nowait(ActorRequest(cmd=cmd, args=args or {}, fut=fut))
        except queue.Full:
            fut.set_result({"ok": False, "error": "actor queue full"})

# ------------------------- MotorDB：方法 → Actor -------------------------
class MotorDB:
    def __init__(self, actor: AirbotActor):
        self.actor = actor

    async def call(self, loop: asyncio.AbstractEventLoop, method: str, params: Dict[str, Any], timeout: float = 3.0) -> Dict[str, Any]:
        cmd, args = MethodRouter.to_actor_cmd(method, params)
        fut = loop.create_future()
        self.actor.submit(cmd, args, fut)
        try:
            res = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            res = {"ok": False, "error": "timeout"}
        return res

# ------------------------- 方法路由 -------------------------
def normalize_message(msg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if "type" not in msg:
        raise ValueError("missing type")
    raw_t = str(msg.get("type", "")).strip()
    t = raw_t.upper()
    mid = msg.get("motor_id")
    if t == "PING":      return "system.ping", {}
    if t == "INIT":
        hz = float(msg.get("spin_freq", 200.0))
        return "system.set_hz", {"spin_freq": hz}
    if t == "UNINIT":
        return "system.unset_hz", {}
    if t == "ENABLE":
        if mid is None: raise ValueError("missing param 'motor_id'")
        return "motor.enable", {"id": int(mid), "enabled": True}
    if t == "DISABLE":
        if mid is None: raise ValueError("missing param 'motor_id'")
        return "motor.enable", {"id": int(mid), "enabled": False}
    if t == "SETZERO":
        if mid is None: raise ValueError("missing param 'motor_id'")
        return "motor.set_zero", {"id": int(mid)}
    if t == "CSP":
        if mid is None or "position" not in msg: raise ValueError("missing motor_id/position")
        return "motor.csp", {"id": int(mid), "position": float(msg["position"])}
    if t == "CSV":
        if mid is None or "velocity" not in msg: raise ValueError("missing motor_id/velocity")
        return "motor.csv", {"id": int(mid), "velocity": float(msg["velocity"])}
    if t == "MIT":
        if mid is None: raise ValueError("missing param 'motor_id'")
        for k in ("q_ref","dq_ref","kp","kd"):
            if k not in msg: raise ValueError(f"missing param '{k}'")
        return "motor.mit", {
            "id": int(mid), "q_ref": float(msg["q_ref"]), "dq_ref": float(msg["dq_ref"]),
            "kp": float(msg["kp"]), "kd": float(msg["kd"]),
            "tau_ff": float(msg.get("tau_ff", 0.0)),
        }
    if t == "PVT":
        if mid is None or "q_ref" not in msg or "dq_ref" not in msg:
            raise ValueError("missing motor_id/q_ref/dq_ref")
        return "motor.pvt", {"id": int(mid), "q_ref": float(msg["q_ref"]), "dq_ref": float(msg["dq_ref"])}
    if t == "SETMODE":
        if mid is None or "mode" not in msg: raise ValueError("missing motor_id/mode")
        return "motor.set_mode", {"id": int(mid), "mode": msg["mode"]}

    if t == "SETPARAM":
        if mid is None: raise ValueError("missing param 'motor_id'")
        name = str(msg.get("name","")).strip().lower()
        val  = msg.get("value")
        if name in ("slew","slew_rate","slewrate"):
            if val is None: raise ValueError("missing param 'value'")
            return "motor.set_slew", {"id": int(mid), "rate": float(val)}
        return "system.noop", {}
    if t in ("PERSISTPARAM","GETPARAM","GETPARAMS","RESETERROR","UPDATE"):
        return "system.noop", {}

    if t in ("GETSTATE","GET_STATE"):
        return "motor.get_state", {"id": int(mid)} if mid is not None else {}
    raise ValueError(f"unknown type: {msg.get('type')}")

class MethodRouter:
    @staticmethod
    def to_actor_cmd(method: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        m = method.strip()
        if m == "motor.csp":
            _req(params, "id"); _req(params, "position")
            return "csp", {"joint": int(params["id"]), "position": float(params["position"])}
        if m == "motor.csv":
            _req(params, "id"); _req(params, "velocity")
            return "csv", {"joint": int(params["id"]), "velocity": float(params["velocity"])}
        if m == "motor.mit":
            for k in ("id","q_ref","dq_ref","kp","kd"): _req(params, k)
            return "mit", {
                "joint": int(params["id"]),
                "q_ref": float(params["q_ref"]), "dq_ref": float(params["dq_ref"]),
                "kp": float(params["kp"]), "kd": float(params["kd"]),
                "tau_ff": float(params.get("tau_ff", 0.0)),
            }
        if m == "motor.pvt":
            for k in ("id","q_ref","dq_ref"): _req(params, k)
            return "pvt", {"joint": int(params["id"]), "q_ref": float(params["q_ref"]), "dq_ref": float(params["dq_ref"])}

        if m == "motor.set_mode":
            _req(params, "id"); _req(params, "mode")
            return "set_mode", {"joint": int(params["id"]), "mode": params["mode"]}
        if m == "motor.enable":
            _req(params, "id"); _req(params, "enabled")
            return "enable", {"joint": int(params["id"]), "enabled": bool(params["enabled"])}
        if m == "motor.set_zero":
            _req(params, "id"); return "set_zero", {"joint": int(params["id"])}
        if m == "motor.set_slew":
            _req(params, "id"); _req(params, "rate")
            return "set_slew", {"joint": int(params["id"]), "rate": float(params["rate"])}
        if m == "motor.get_state":
            return "get_state", {"joint": int(params["id"])} if "id" in params else {}

        if m == "motor.batch":
            _req(params, "items"); return "batch", {"items": params["items"]}

        if m == "system.ping":    return "ping", {}
        if m == "system.reset":   return "reset", {}
        if m == "system.quit":    return "quit", {}
        if m == "system.set_hz":  return "set_hz", {"spin_freq": float(params.get("spin_freq", 200.0))}
        if m == "system.unset_hz":return "unset_hz", {}
        if m == "system.noop":    return "ping", {}

        if m in ("csp","csv","mit","pvt","set_mode","enable","set_zero","set_slew","get_state","batch","ping","reset","quit","set_hz","unset_hz"):
            return m, params
        raise ValueError(f"unknown method: {method}")

def _req(d: Dict[str, Any], k: str):
    if k not in d:
        raise ValueError(f"missing param '{k}'")

# ------------------------- JSON-Lines Server -------------------------
class JsonLineServer:
    def __init__(self, motor, host, port, request_timeout=3.0):
        self.motor = motor
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self._wlock = asyncio.Lock()

    async def handle_conn(self, reader, writer):
        peer = writer.get_extra_info("peername")
        print(f"[conn] {peer} connected")
        try:
            while True:
                line = await reader.readline()
                if not line: break
                line = line.strip()
                if not line: continue
                asyncio.create_task(self._serve_one(line, writer))
        finally:
            try:
                writer.close(); await writer.wait_closed()
            except Exception: pass
            print(f"[conn] {peer} closed")

    async def _serve_one(self, line: bytes, writer):
        try:
            raw = json.loads(line.decode("utf-8"))
            req_id = raw.get("req_id")
            method, params = normalize_message(raw)

            t_upper = str(raw.get("type","")).upper()
            if t_upper == "GETPARAM":
                res = {"ok": True, "data": {"value": None}}
            elif t_upper == "GETPARAMS":
                res = {"ok": True, "data": {}}
            else:
                loop = asyncio.get_running_loop()
                res = await self.motor.call(loop, method, params, timeout=self.request_timeout)

            if req_id is not None:
                res = {**res, "req_id": req_id}

            async with self._wlock:
                writer.write((json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
        except Exception as e:
            err = {"ok": False, "error": f"bad request: {e}"}
            async with self._wlock:
                writer.write((json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()

    async def run(self):
        server = await asyncio.start_server(self.handle_conn, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in (server.sockets or []))
        print(f"[server] listening on {addrs}")
        async with server:
            await server.serve_forever()

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ctrl-hz", type=float, default=200.0)
    ap.add_argument("--request-timeout", type=float, default=2.0)
    args = ap.parse_args()

    actor = AirbotActor(headless=args.headless, ctrl_hz=args.ctrl_hz)  
    actor.start()

    motor = MotorDB(actor)
    server = JsonLineServer(motor, host=args.host, port=args.port, request_timeout=args.request_timeout)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("[main] Ctrl-C, stopping...")
    finally:
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        actor.submit("quit", {}, fut)
        actor.join(timeout=2.0)
        print("[main] bye")

if __name__ == "__main__":
    main()