#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import math
import sys
from typing import List

from discoverse.examples.motor.client.airbotplay import Play
from discoverse.examples.motor.client.motor_client import MotorType, EEFType


HOST = "127.0.0.1"
PORT = 8890

# 收敛阈值
TOL_POS = 0.05      # rad
TOL_VEL = 0.05      # rad/s

# 关节与夹爪限幅（弧度）
MIN_Q = [-3.1416, -2.9671, -0.087266, -3.0107, -1.7628, -3.0107]
MAX_Q = [2.0944, 0.17453, 3.1416, 3.0107, 1.7628, 3.0107]
GRIP_MIN, GRIP_MAX = 0.0, 0.072

# server 端速度上限（rad/s）
JOINT_V_MAX = math.pi / 5


def _check_targets(joints: List[float], gripper: float) -> bool:
    if len(joints) != 6:
        return False
    for idx, q in enumerate(joints):
        if not (MIN_Q[idx] <= q <= MAX_Q[idx]):
            return False
    return GRIP_MIN <= gripper <= GRIP_MAX


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
        auto_reconnect=True,
    )


async def _ensure_ready() -> Play:
    play = _create_play()
    await play.connect()
    return play


async def _wait_until_reached(play: Play, targets: List[float]) -> None:
    """等待到达目标位置"""
    initial_state = await play.state()
    print(initial_state)
    if initial_state.is_valid:
        print("state:", [round(p, 4) for p in initial_state.pos])
    while True:
        state = await play.state()
        if not state.is_valid:
            await asyncio.sleep(0.02)
            continue
        pos = state.pos
        vel = state.vel
        # print("pos:", [round(p, 4) for p in pos], "vel:", [round(v, 4) for v in vel])
        pos_ok = all(abs(p - t) < TOL_POS for p, t in zip(pos, targets))
        vel_ok = all(abs(v) < TOL_VEL for v in vel)
        if pos_ok and vel_ok:
            print("到位")
            return
        await asyncio.sleep(0.01)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Play API 示范：一次位置下发 + 到位轮询")
    parser.add_argument("-j", "--joints", nargs=6, type=float, required=True, help="6 个关节目标（弧度）")
    parser.add_argument("-g", "--gripper", type=float, default=0.0, help="夹爪目标（默认 0.0）")
    args = parser.parse_args()

    if not _check_targets(args.joints, args.gripper):
        print("参数超出阈值，请检查 6 关节与夹爪范围。", file=sys.stderr)
        return 1

    play = await _ensure_ready()

    try:
        if not await play.init(interface="can0", spin_freq=250.0):
            print("初始化失败", file=sys.stderr)
            return 1
        if not await play.enable():
            print("电机使能失败", file=sys.stderr)
            return 1
        # if not await play.set_param("slew_rate", JOINT_V_MAX):
        #     print("设定速度上限失败", file=sys.stderr)
        #     return 1

        targets = list(args.joints)
        if play.has_eef:
            targets.append(float(args.gripper))

        # 为所有关节设置速度限制为 math.pi/10
        max_velocities = [JOINT_V_MAX] * len(targets)
        print(f"下发目标位置: {[round(t, 7) for t in targets]}，速度上限: {[round(v, 7) for v in max_velocities]}") 
        if not await play.pvt(targets, max_vel=max_velocities):
            print("下发 PVT 指令失败", file=sys.stderr)
            return 1

        await _wait_until_reached(play, targets)
        print("untill reached targets:",await play.state())
        return 0
    finally:
        await play.disable()
        await play.uninit()
        await play.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)