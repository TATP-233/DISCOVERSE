#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import math
import sys
import time
from typing import List

from discoverse.examples.motor.client.airbotplay import Play
from discoverse.examples.motor.client.motor_client import MotorType, EEFType


HOST = "127.0.0.1"
PORT = 8890

CTRL_HZ = 250.0
DT = 1.0 / CTRL_HZ

SLEW_MAX = math.pi / 10  # rad/s
TOL_POS = 0.1
TOL_VEL = 0.1
SWING_AMPLITUDE = 2.0  # rad


def _create_play() -> Play:
    return Play.create(
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        MotorType.EC,
        EEFType.G2,  # eef_type (will be determined)
        MotorType.EC,  # eef_motor_type
        host=HOST,
        port=PORT,
    )


async def _ensure_ready() -> Play:
    play = _create_play()
    await play.connect()
    return play


async def _home_to_zero(play: Play) -> None:
    joint_count = play.joint_count
    target_count = play.total_count
    targets = [0.0] * target_count

    await play.set_param("slew_rate", SLEW_MAX)
    await play.pvt(targets)

    while True:
        state = await play.state()
        if not state.is_valid:
            await asyncio.sleep(0.02)
            continue
        pos = state.pos
        vel = state.vel
        pos_ok = all(abs(p) < TOL_POS for p in pos[:joint_count])
        vel_ok = all(abs(v) < TOL_VEL for v in vel[:joint_count])
        eef_ok = True
        if play.has_eef and target_count > joint_count:
            eef_pos = pos[joint_count:target_count]
            eef_vel = vel[joint_count:target_count]
            eef_ok = all(abs(p) < TOL_POS for p in eef_pos) and all(abs(v) < TOL_VEL for v in eef_vel)
        if pos_ok and vel_ok and eef_ok:
            print("Arrived at zero position")
            return
        await asyncio.sleep(0.01)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Play API 示范：关节 1 简单摆动")
    parser.add_argument("-T", type=float, required=True, help="摆动周期（秒），建议 2~20")
    args = parser.parse_args()

    period = float(args.T)
    if not (2.0 <= period <= 20.0):
        print("参数 T 不合理，推荐范围 2〜20 秒。", file=sys.stderr)
        return 1

    play = await _ensure_ready()

    try:
        if not await play.init(interface="can0", spin_freq=CTRL_HZ):
            print("初始化失败", file=sys.stderr)
            return 1
        if not await play.enable():
            print("电机使能失败", file=sys.stderr)
            return 1

        await _home_to_zero(play)

        if not await play.set_param("slew_rate", SLEW_MAX):
            print("设定速度上限失败", file=sys.stderr)
            return 1

        print("开始关节 1 摆动（Ctrl+C 结束）")
        t0 = time.time()
        while True:
            t = time.time() - t0
            q1 = SWING_AMPLITUDE * math.sin(2.0 * math.pi * (t / period))
            targets = [q1] + [0.0] * (play.joint_count - 1)
            if play.has_eef:
                targets.append(0.0)
            ok = await play.pvt(targets)
            if not ok:
                print("PVT 指令失败，尝试继续...", file=sys.stderr)
                await asyncio.sleep(DT)
                continue
            await asyncio.sleep(DT)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备停止。")
    finally:
        await play.disable()
        await play.uninit()
        await play.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)

