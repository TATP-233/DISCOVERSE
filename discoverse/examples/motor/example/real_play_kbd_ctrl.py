#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example application of controlling the simulated robot arm using keyboard input (CSV velocity control).

仿照官方示例的键盘控制逻辑，在本项目的仿真器上通过 CSV 模式控制关节速度。
"""

import argparse
import asyncio
import curses
import logging
from typing import List

from discoverse.examples.motor.client.airbotplay import Play
from discoverse.examples.motor.client.motor_client import MotorType, EEFType


LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)-8s] "
    "[%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("kbd_csv")


HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8890


def _create_play(host: str, port: int) -> Play:
    """
    创建与 `real_play_pos_ctrl.py` 相同配置的 Play 实例：
    - 6 个关节：EC 电机
    - 末端执行器：G2 夹爪 + EC 电机
    """
    return Play.create(
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        EEFType.G2,
        MotorType.EC,
        host=host,
        port=port,
        auto_reconnect=True,
    )


async def _setup_play(host: str, port: int) -> Play:
    """连接仿真器、初始化并使能所有电机。"""
    play = _create_play(host, port)
    await play.connect()
    logger.info("Connected to motor server at %s:%d", host, port)

    if not await play.init(interface="can0", spin_freq=200.0):
        raise RuntimeError("Init failed")
    logger.info("Init OK")

    if not await play.enable():
        raise RuntimeError("Enable failed")
    logger.info("Enable OK")

    return play


async def _teardown_play(play: Play) -> None:
    """关闭前的收尾：停机、uninit、断开连接。"""
    try:
        await play.disable()
        await play.uninit()
    finally:
        await play.close()
        logger.info("Disconnected")


async def _csv_loop(stdscr, play: Play, speed: float) -> None:
    """
    主循环：从键盘读取按键，并通过 CSV 模式发送关节速度指令。

    约定：
    - 数字键 1~0、'-', '=' 分别控制 6 个关节的正负向转动
    - 'x'：立即停止所有关节（速度清零）
    - 'z'：退出程序
    """
    stdscr.nodelay(True)
    curses.curs_set(0)

    help_lines = [
        "Keyboard CSV control (simulator):",
        "  1/2: +joint1 / -joint1",
        "  3/4: +joint2 / -joint2",
        "  5/6: +joint3 / -joint3",
        "  7/8: +joint4 / -joint4",
        "  9/0: +joint5 / -joint5",
        "  -/=: +joint6 / -joint6",
        "  x : stop all joints (zero velocity)",
        "  [/]: close / open gripper (EEF) in velocity mode",
        "  z : quit",
        f"  current speed magnitude: {speed:.3f} rad/s",
    ]

    for i, line in enumerate(help_lines):
        stdscr.addstr(i, 0, line)
    stdscr.refresh()

    try:
        while True:
            # 默认本周期所有关节速度为 0（不按键就不动）
            velocities: List[float] = [0.0] * 7  # 6 关节 + 1 夹爪占位
            key = stdscr.getch()

            if key != -1:
                # 有按键时，只在当前周期对对应关节施加速度
                if key == ord("1"):
                    velocities[0] = +speed
                elif key == ord("2"):
                    velocities[0] = -speed
                elif key == ord("3"):
                    velocities[1] = +speed
                elif key == ord("4"):
                    velocities[1] = -speed
                elif key == ord("5"):
                    velocities[2] = +speed
                elif key == ord("6"):
                    velocities[2] = -speed
                elif key == ord("7"):
                    velocities[3] = +speed
                elif key == ord("8"):
                    velocities[3] = -speed
                elif key == ord("9"):
                    velocities[4] = +speed
                elif key == ord("0"):
                    velocities[4] = -speed
                elif key == ord("-"):
                    velocities[5] = +speed
                elif key == ord("="):
                    velocities[5] = -speed
                elif key == ord("["):
                    velocities[6] = -speed  # gripper close
                elif key == ord("]"):
                    velocities[6] = +speed  # gripper open
                elif key == ord("x"):
                    velocities = [0.0] * 7
                elif key == ord("z"):
                    break

            # 发送当前速度命令（6 关节 + 1 夹爪占位）
            await play.csv(velocities)

            # 稍作延时，避免占满 CPU，也给仿真器一点时间
            await asyncio.sleep(0.02)
    finally:
        # 停机
        try:
            await play.csv([0.0] * 7)
        except Exception:
            pass


def _curses_entry(stdscr, host: str, port: int, speed: float) -> None:
    """curses.wrapper 调用的同步入口，内部运行异步逻辑。"""

    async def runner():
        play = await _setup_play(host, port)
        try:
            await _csv_loop(stdscr, play, speed)
        finally:
            await _teardown_play(play)

    asyncio.run(runner())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keyboard CSV control on simulator (uses motor JSON-line server)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST_DEFAULT,
        help="motor JSON-line server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT_DEFAULT,
        help="motor JSON-line server port (default: 8890)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.2,
        help="joint velocity magnitude in rad/s (default: 0.5)",
    )
    args = parser.parse_args()

    logger.info(
        "Starting keyboard CSV control: host=%s port=%d speed=%.3f",
        args.host,
        args.port,
        args.speed,
    )

    curses.wrapper(_curses_entry, args.host, args.port, args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


