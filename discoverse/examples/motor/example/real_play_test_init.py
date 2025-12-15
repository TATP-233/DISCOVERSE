#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import math
import time
from typing import List

from discoverse.examples.motor.client.airbotplay import Play
from discoverse.examples.motor.client.motor_client import MotorType, EEFType


# Configuration
HOST = "127.0.0.1"
PORT = 8890
CAN_INTERFACE = "can0"


def create_play(host: str = HOST, port: int = PORT) -> Play:
    """
    Create Play object with G2 gripper.
    """
    return Play.create(
        MotorType.OD,
        MotorType.OD,
        MotorType.OD,
        MotorType.DM,
        MotorType.DM,
        MotorType.DM,
        EEFType.G2,  # eef_type (will be determined)
        MotorType.DM,  # eef_motor_type
        host=host,
        port=port,
        auto_reconnect=True,
    )




async def test_pvt_control(play: Play, test_name: str):
    """
    Test PVT control mode.
    """
    print(f"\n[TEST] {test_name}")
    print("[STEP] Attempting to send PVT command...")
    
    joint_positions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    gripper_position = 0.0
    
    target_position = joint_positions + [gripper_position]
    velocity = [math.pi / 10] * 7
    
    try:
        result = await play.pvt(target_position, max_vel=velocity)
        if result:
            print(f"[SUCCESS] PVT command sent successfully")
            # Get state to verify
            state = await play.state()
            if state.is_valid:
                print(f"[INFO] Start position: {state.pos}")
            else:
                print("[WARN] State is not valid")
        else:
            print(f"[FAILED] PVT command failed")
    except Exception as e:
        print(f"[ERROR] Exception during PVT command: {e}")


async def test_pvt_control_again(play: Play, test_name: str):
    """
    Test PVT control mode.
    """
    print(f"\n[TEST] {test_name}")
    print("[STEP] Attempting to send PVT command...")
    
    joint_positions = [-0.2, -0.3, -0.3, -0.1, -0.1, 1]
    gripper_position = 0.07
    
    target_position = joint_positions + [gripper_position]
    velocity = [math.pi / 10] * 7
    
    try:
        result = await play.pvt(target_position, max_vel=velocity)
        if result:
            print(f"[SUCCESS] PVT command sent successfully")
            # Get state to verify
            state = await play.state()
            if state.is_valid:
                print(f"[INFO] Current position: {state.pos}")
            else:
                print("[WARN] State is not valid")
        else:
            print(f"[FAILED] PVT command failed")
    except Exception as e:
        print(f"[ERROR] Exception during PVT command: {e}")


async def main():
    """
    Test sequence:
    1. Connect to server and use Init and PVT control mode
    2. Use uninit function
    3. Test PVT control mode again (should fail)
    4. Disconnect all connections
    """
    print("=" * 60)
    print("Test: Init -> PVT -> Uninit -> PVT (should fail) -> Close")
    print("=" * 60)
    
    # Step 1: Create Play instance with gripper
    print("\n[STEP 1] Connecting to motor RPC server...")
    
    play = create_play(host=HOST, port=PORT)
    await play.connect()
    print("[INFO] Connection established.")
    
    try:
        # Step 2: Initialize motors
        print("\n[STEP 2] Initializing motors...")
        if not await play.init(interface=CAN_INTERFACE, spin_freq=250.0):
            raise RuntimeError("The arm initialization failed.")
        print("[SUCCESS] Motors initialized.")
        
        # Step 3: Enable motors
        print("\n[STEP 3] Enabling motors...")
        if not await play.enable():
            raise RuntimeError("Failed to enable motors.")
        print("[SUCCESS] Motors enabled.")
        
        # Step 4: Test PVT control mode (first time - should work)
        await test_pvt_control(play, "First PVT test (after init)")
        await asyncio.sleep(5)  # Wait a bit
        state = await play.state()
        if state.is_valid:
            print(f"[INFO] End position: {state.pos}")
        
        # Step 5: Disable motors before uninit
        # print("\n[STEP 5] Disabling motors...")
        # await play.disable()
        # print("[SUCCESS] Motors disabled.")
        
        # Step 6: Uninitialize motors
        print("\n[STEP 6] Uninitializing motors...")
        uninit_result = await play.uninit()
        if uninit_result:
            print("[SUCCESS] Motors uninitialized.")
        else:
            print("[WARN] Uninit returned False, but continuing...")
        
        # Step 7: Test PVT control mode again (should fail after uninit)
        
        try:
            await test_pvt_control_again(play, "Second PVT test (after uninit - should fail)")
            await asyncio.sleep(5)
            state = await play.state()
            if state.is_valid:
                print(f"[INFO] Second End position: {state.pos}")
        except Exception as e:
            print(f"[EXPECTED ERROR] PVT command failed as expected after uninit: {e}")
        try:
            print("[INFO] Try to ping motor after uninit:",await play.ping())
        except Exception as e:
            print(f"[EXPECTED ERROR] Ping failed as expected after uninit: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] 检测到Ctrl+C，正在停止...")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
    finally:
        # Step 8: Close connection
        print("\n[STEP 8] Closing RPC channel...")
        await play.init()
        await play.close()
        print("[SUCCESS] Session closed.")
        print("\n" + "=" * 60)
        print("Test completed.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

