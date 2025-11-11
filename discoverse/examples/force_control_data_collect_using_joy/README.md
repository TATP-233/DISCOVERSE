# Robot Arm Joystick Controller

This project implements joystick control for a robot arm using a fixed orientation approach. The end-effector maintains a downward-pointing orientation while the position is controlled via an Xbox controller.

## Architecture

The system integrates several components:

1. **JoyController** (`joy/joy_controller.py`): Pygame-based joystick interface
2. **Mink_IK** (`../mocap_ik/mink_arm_ik.py`): Inverse kinematics solver
3. **ImpedanceController** (`../force_control/impedance_control.py`): Robot control interface
4. **AirbotPlay** (`airbot_py`): Robot hardware communication

## Control Flow

1. Joystick input → position delta
2. Update target end-effector position
3. Solve inverse kinematics with fixed orientation
4. Send joint targets to robot via AirbotPlay

## Files

- `robot_joy_controller.py`: Main robot control implementation
- `test_joy_only.py`: Testing script without robot hardware
- `joy/`: Joystick controller module
- `arm_controller.py`: (existing file)

## Usage

### Testing (without robot)

```bash
python test_joy_only.py
```

This allows testing joystick input and IK solving without requiring robot hardware.

### Full Robot Control

```bash
python robot_joy_controller.py
```

**Requirements:**
- Xbox controller connected
- Robot properly connected and configured
- Required dependencies installed

### Controls

- **Left Stick**: Move in X-Y plane
- **Left Trigger (LT)**: Move down (Z-)
- **Right Trigger (RT)**: Move up (Z+)
- **A Button**: Reset to starting position
- **START Button**: Quit

## Configuration

### Fixed Orientation
The robot maintains a fixed downward-pointing orientation:
```python
fixed_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # qw, qx, qy, qz
```

### Position Limits
Safety limits are enforced:
- X: 0.1 to 0.4 meters
- Y: -0.3 to 0.3 meters
- Z: 0.1 to 0.4 meters

### Movement Parameters
- Position step: 1mm per joystick input
- Deadzone: 0.1 (10% of full range)
- Robot velocity: 0.1 rad/s
- Robot acceleration: 0.1 rad/s²

## Dependencies

```bash
pip install pygame
pip install /path/to/airbot_py-5.1.6-py3-none-any.whl
```

Additional dependencies are managed by the DISCOVERSE environment.

## Implementation Notes

### Thread Architecture
- Main thread: Joystick event processing and UI
- Background thread: Robot state monitoring (100Hz)

### IK Solver
- Uses Mink IK with end-effector frame task
- Position cost: 100.0
- Orientation cost: 10.0
- Convergence thresholds: 1e-3 for position and orientation

### Robot Interface
- Uses AirbotPlay for high-level robot control
- Position control with motion planning enabled
- Non-blocking commands for smooth operation

## Safety Features

- Position limits enforced before IK solving
- IK convergence checking before sending commands
- Graceful error handling and recovery
- Emergency stop via START button

## Troubleshooting

### "No joystick found"
- Ensure Xbox controller is connected and recognized
- Check USB connection
- Try unplugging and reconnecting

### "IK failed to converge"
- Target position may be unreachable
- Check position limits
- Try resetting to starting position (A button)

### Robot communication errors
- Verify robot is powered and connected
- Check network/USB connection to robot
- Ensure AirbotPlay service is running