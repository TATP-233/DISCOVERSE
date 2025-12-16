#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Play API 输出类型测试脚本。

执行顺序参考其他 real_* 示例，通过打印每个接口的返回类型/值，
并尝试在不同状态下重复调用来观察布尔方法的变化。
"""

import argparse
import asyncio
from typing import Any, Callable, Awaitable, Optional, Sequence

# 添加项目根目录到 sys.path，以便使用相对导入
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from client.airbotplay import Play , PlayState
from client.motor_client import MotorType, EEFType


HOST = "127.0.0.1"
PORT = 8890
DEFAULT_INTERFACE = "can0"
DEFAULT_SPIN_FREQ = 200.0


def _create_play(host: str, port: int) -> Play:
    return Play.create(
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        EEFType.G2,  # eef_type (will be determined)
        MotorType.EC,  # eef_motor_type
        # host=host,
        # port=port,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, PlayState):
        return f"PlayState(is_valid={value.is_valid}, count={value.count})"
    if isinstance(value, list):
        summary = f"list(len={len(value)})"
        return summary
    if isinstance(value, Play):
        return f"Play(joints={value.joint_count}, total={value.total_count})"
    return repr(value)


async def _report_async(name: str, fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
    try:
        result = await fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 调试输出
        print(f"[{name}] raised {exc.__class__.__name__}: {exc}")
        return None
    else:
        print(f"[{name}] -> type={type(result).__name__}, value={_format_value(result)}")
        return result


async def test_play_interfaces(
    host: str = HOST,
    port: int = PORT,
    *,
    interface: str = DEFAULT_INTERFACE,
    spin_freq: float = DEFAULT_SPIN_FREQ,
) -> None:
    """
    在真实设备上依次调用 Play 的关键接口并打印输出类型。
    """

    play = _create_play(host, port)
    print(f"[create] -> type={type(play).__name__}, value={_format_value(play)}")
    await play.connect()
    print("[connect] OK")

    try:
        await _report_async("ping (before init)", play.ping)
        await _report_async("state (before init)", play.state)
        print("[state type]: ",await play.state())
        print("---- 开始初始化后测试 ----")
        # 尝试在未初始化状态下使能，观察布尔返回
        await _report_async("enable (pre-init)", play.enable)

        await _report_async("init", play.init, interface=interface, spin_freq=spin_freq)
        await _report_async("ping (after init)", play.ping)

        await _report_async("enable", play.enable)
        await _report_async("set_zero", play.set_zero)

        target_count = play.total_count
        zero_targets: Sequence[float] = [0.0] * target_count
        await _report_async("pvt (zeros)", play.pvt, zero_targets)

        # 故意传入无效长度，观察异常信息
        await _report_async("pvt (invalid length)", play.pvt, [0.0])

        await _report_async("state (after pvt)", play.state)
        print("[state type]: ",await play.state())

        await _report_async("disable", play.disable)
        # 再次调用 disable，观察布尔返回是否一致
        await _report_async("disable (again)", play.disable)

        await _report_async("uninit", play.uninit)
        await _report_async("uninit (again)", play.uninit)
    finally:
        await play.close()
        print("[close] OK")


async def _async_main(args: argparse.Namespace) -> int:
    try:
        await test_play_interfaces(
            host=args.host,
            port=args.port,
            interface=args.interface,
            spin_freq=args.spin_freq,
        )
    except KeyboardInterrupt:
        print("\n终止测试。")
        return 130
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 Play API 接口输出类型")
    parser.add_argument("--host", default=HOST, help="控制器 host")
    parser.add_argument("--port", type=int, default=PORT, help="控制器端口")
    parser.add_argument("--interface", default=DEFAULT_INTERFACE, help="Motor init 接口名")
    parser.add_argument("--spin-freq", type=float, default=DEFAULT_SPIN_FREQ, help="init spin_freq 参数")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())


