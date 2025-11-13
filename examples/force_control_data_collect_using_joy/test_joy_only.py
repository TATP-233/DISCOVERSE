#!/usr/bin/env python3
"""
Test script for joystick functionality without requiring robot hardware.
This allows testing the joystick input mapping and IK solving.
"""
import os
import sys
import time
import numpy as np

# Add paths for imports
sys.path.append('/home/lue/qiuzhi/DISCOVERSE/discoverse/examples/force_control_data_collect_using_joy/joy')
sys.path.append('/home/lue/qiuzhi/DISCOVERSE/discoverse/examples/mocap_ik')

from joy_controller import JoyController
from mink_arm_ik import Mink_IK
from discoverse import DISCOVERSE_ASSETS_DIR

class JoyTestController:
    def __init__(self):
        # Initialize joystick
        self.joy = JoyController()
        
        # Initialize IK solver (optional - comment out if not needed for testing)
        try:
            mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator", "robot_airbot_play_force.xml")
            self.ik_solver = Mink_IK(mjcf_path, arm_dof=6)
            self.ik_available = True
        except Exception as e:
            print(f"IK solver not available: {e}")
            self.ik_available = False
        
        # Control state
        self.running = False
        
        # Fixed orientation (pointing downward)
        self.fixed_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # qw, qx, qy, qz
        
        # Current end-effector position
        self.current_position = np.array([0.205, 0.0, 0.22])  # Starting position
        self.current_qpos = np.zeros(6)
        
        # Movement parameters
        self.position_step = 0.001  # 1mm per step
        self.deadzone = 0.1
        
        # Continuous movement variables
        self.continuous_move_x = 0
        self.continuous_move_y = 0
        self.continuous_move_z = 0
        
        self._setup_joy_callbacks()
        
    def _setup_joy_callbacks(self):
        """Setup joystick button and axis callbacks"""
        # Left stick X-axis (axis 0) for X movement
        self.joy.add_axis_callback(0, self._on_x_axis)
        
        # Left stick Y-axis (axis 1) for Y movement  
        self.joy.add_axis_callback(1, self._on_y_axis)
        
        # Left trigger (axis 2) for Z down
        self.joy.add_axis_callback(2, self._on_z_down)
        
        # Right trigger (axis 5) for Z up  
        self.joy.add_axis_callback(5, self._on_z_up)
        
        # Start button (7) to quit
        self.joy.add_button_down_callback(7, self._quit)
        
        # A button (0) to reset position
        self.joy.add_button_down_callback(0, self._reset_position)
        
    def _on_x_axis(self, value):
        """Handle X-axis movement - store continuous value"""
        if abs(value) > self.deadzone:
            normalized_value = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
            if value < 0:
                normalized_value = -normalized_value
            self.continuous_move_x = normalized_value
        else:
            self.continuous_move_x = 0
            
    def _on_y_axis(self, value):
        """Handle Y-axis movement - store continuous value"""
        if abs(value) > self.deadzone:
            normalized_value = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
            if value < 0:
                normalized_value = -normalized_value
            self.continuous_move_y = -normalized_value
        else:
            self.continuous_move_y = 0
            
    def _on_z_down(self, value):
        """Handle Z down movement - store continuous value"""
        trigger_value = (value + 1) / 2
        if trigger_value > self.deadzone:
            self.continuous_move_z = -trigger_value
        elif self.continuous_move_z < 0:
            self.continuous_move_z = 0
            
    def _on_z_up(self, value):
        """Handle Z up movement - store continuous value"""
        trigger_value = (value + 1) / 2
        if trigger_value > self.deadzone:
            self.continuous_move_z = trigger_value
        elif self.continuous_move_z > 0:
            self.continuous_move_z = 0
            
    def _move_position(self, axis, delta):
        """Move position along specified axis and test IK"""
        self.current_position[axis] += delta
        
        # Safety limits
        self.current_position[0] = np.clip(self.current_position[0], 0.1, 0.4)  # X limits
        self.current_position[1] = np.clip(self.current_position[1], -0.3, 0.3)  # Y limits  
        self.current_position[2] = np.clip(self.current_position[2], 0.1, 0.4)  # Z limits
        
        self._test_ik()
        
    def _update_continuous_movement(self, dt):
        """Update position based on continuous joystick input values"""
        if abs(self.continuous_move_x) > 0 or abs(self.continuous_move_y) > 0 or abs(self.continuous_move_z) > 0:
            # Apply movement with frame time
            dx = self.continuous_move_x * self.position_step * dt * 60  # Scale by 60 for smoother movement
            dy = self.continuous_move_y * self.position_step * dt * 60
            dz = self.continuous_move_z * self.position_step * dt * 60
            
            old_position = self.current_position.copy()
            
            self.current_position[0] += dx
            self.current_position[1] += dy
            self.current_position[2] += dz
            
            # Safety limits
            self.current_position[0] = np.clip(self.current_position[0], 0.1, 0.4)  # X limits
            self.current_position[1] = np.clip(self.current_position[1], -0.3, 0.3)  # Y limits  
            self.current_position[2] = np.clip(self.current_position[2], 0.1, 0.4)  # Z limits
            
            # Only update IK if position actually changed
            if not np.allclose(old_position, self.current_position, atol=1e-6):
                self._test_ik()
        
    def _test_ik(self):
        """Test IK solving and print results"""
        if not self.ik_available:
            print(f"Position: [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}] (IK not available)")
            return
            
        try:
            # Solve inverse kinematics
            target_joints, converged = self.ik_solver.solve_ik(
                target_pos=self.current_position,
                target_ori=self.fixed_orientation,
                current_qpos=self.current_qpos
            )
            
            if converged:
                self.current_qpos = target_joints.copy()
                print(f"Position: [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}] -> Joints: {target_joints}")
            else:
                print(f"Position: [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}] -> IK FAILED")
                
        except Exception as e:
            print(f"Error in IK: {e}")
            
    def _reset_position(self):
        """Reset to starting position"""
        self.current_position = np.array([0.205, 0.0, 0.22])
        self.current_qpos = np.zeros(6)
        self._test_ik()
        print("Position reset")
        
    def _quit(self):
        """Quit the application"""
        print("Shutting down...")
        self.running = False
        
    def start(self):
        """Start the test controller"""
        print("Starting Joy Test Controller")
        print(f"Controller: {self.joy.get_info()['name']}")
        print(f"Starting position: {self.current_position}")
        print(f"Fixed orientation (qw,qx,qy,qz): {self.fixed_orientation}")
        print(f"IK available: {self.ik_available}")
        print("\nControls:")
        print("- Left stick: Move in X-Y plane")
        print("- LT: Move down (Z-)")
        print("- RT: Move up (Z+)")
        print("- A: Reset position")
        print("- START: Quit")
        print("-" * 50)
        
        # Test initial IK
        self._test_ik()
        
        # Main control loop
        self.running = True
        last_time = time.time()
        try:
            while self.running:
                current_time = time.time()
                dt = current_time - last_time
                
                self.joy.process_events()
                self._update_continuous_movement(dt)
                
                # Limit to ~60 FPS
                if dt < 1/60:
                    time.sleep(1/60 - dt)
                
                last_time = current_time
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            
        finally:
            self.joy.quit()
            print("Test complete")

if __name__ == "__main__":
    try:
        controller = JoyTestController()
        controller.start()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have an Xbox controller connected!")