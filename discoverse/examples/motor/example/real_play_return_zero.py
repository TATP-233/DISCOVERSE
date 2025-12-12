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
TIMEOUT = 10.0


def all_zero(pos: List[float]) -> bool:
    """Check if all positions are close to zero (within 1e-3)"""
    return all(abs(p) < 1e-3 for p in pos)


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


async def send_zero_command(play: Play, timeout: float):
    """
    Stream PVT commands that request zero position.
    """
    joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gripper_position = 0.0
    
    target_position = joint_positions + [gripper_position]
    velocity = [math.pi / 5] * 7

    start = time.time()
    print("[STEP] Streaming PVT commands toward zero position...")
    initial_state = await play.state()
    if initial_state.is_valid:
        print(f"[INFO] Initial position: {initial_state.pos}")
    
    while time.time() - start < timeout:
        await play.pvt(target_position, max_vel=velocity)
        
        state = await play.state()
        if state.is_valid:
            if all_zero(state.pos):
                print("[SUCCESS] All motors reached zero position.")
                return
        await asyncio.sleep(0.004)
    
    print("[TIMEOUT] Motors did not converge to zero within timeout.")


async def main():
    print("[STEP] Connecting to motor RPC server...")
    
    # Create the Play instance with gripper
    play = create_play(host=HOST, port=PORT)
    await play.connect()
    print("[INFO] Connection established.")
    
    try:
        if not await play.init(interface=CAN_INTERFACE, spin_freq=250.0):
            raise RuntimeError("The arm initialization failed.")
        
        if not await play.enable():
            raise RuntimeError("Failed to enable motors.")
        
        # Send zero commands
        await send_zero_command(play, timeout=TIMEOUT)
        
    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在停止机械臂并释放资源...")
    finally:
        print("[STEP] Sending disable commands to all motors...")
        await play.disable()
        
        print("[STEP] Uninitializing motors...")
        await play.uninit()
        
        print("[STEP] Closing RPC channel...")
        await play.close()
        print("[INFO] Session closed.")


if __name__ == "__main__":
    asyncio.run(main())