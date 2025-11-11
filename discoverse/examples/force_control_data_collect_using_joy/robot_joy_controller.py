#!/usr/bin/env python3
import os
import time
import threading
import numpy as np
from discoverse.examples.mocap_ik.mink_arm_ik import Mink_IK
from discoverse.examples.force_control.impedance_control import ImpedanceController
from discoverse.examples.force_control_data_collect_using_joy.joy.joy_controller import JoyController
from airbot_py.arm import AIRBOTPlay, RobotMode
from discoverse import DISCOVERSE_ASSETS_DIR


np.set_printoptions(precision=3, suppress=True, linewidth=500)

class RobotJoyController:
    max_tau = np.array([10.0, 10.0, 10.0, 1.5, 1.5, 1.5]) * 0.5

    def __init__(self):
        # Initialize components
        self.mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator", "robot_airbot_play_force.xml")
        self._init_robot_model()
        self._init_controllers()
        self._init_joystick()
        
        # Robot state
        self.current_qpos = np.zeros(6)
        self.current_qvel = np.zeros(6)
        self.current_tau = np.zeros(6)
        
        # Control state
        self.running = False
        self.state_thread = None
        
        # Fixed orientation (pointing downward)
        # self.fixed_orientation = np.array([-0.707, 0.707, 0.0, 0.0])  # qw, qx, qy, qz
        self.fixed_orientation = np.array([0.707, 0.0, 0.707, 0.0])  # qw, qx, qy, qz
        
        # Current end-effector position
        # self.current_position = np.array([0.725, 0, 0.113])  # Starting position
        self.current_position = np.array([0.259, -0.026, 0.176])  # Starting position
        
        # Movement parameters
        self.position_step = 0.0003  # 0.3mm per step
        self.deadzone = 0.1
        self.coeff = [0.6, 0.6, 0.6, 1.35474, 1.32355, 1.5]
        
        # Continuous movement variables
        self.continuous_move_x = 0
        self.continuous_move_y = 0
        self.continuous_move_z = 0
        
        self.got_init_force = False
        self.init_force = np.zeros(3)

        self.start_collect = False

        self.pause = False
        
    def _init_robot_model(self):
        """Initialize robot model and IK solver"""
        self.ik_solver = Mink_IK(self.mjcf_path, arm_dof=6)

    def _init_controllers(self):
        """Initialize impedance controller and robot interface"""
        import mujoco
        mj_model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        
        # Impedance controller gains
        self.kp = np.array([30, 100, 100, 10, 20, 20]) * 1.0
        self.kd = np.array([50, 50, 50, 2., 2.5, 1.]) * 0.01
        self.impedance_controller = ImpedanceController(mj_model, kpl=self.kp*0.0, kdl=self.kd*0.0)

        # AirbotPlay interface for robot communication
        print("Connecting to AirbotPlay...")
        self.airbot_play = AIRBOTPlay("localhost", 50051)
        self.airbot_play.connect()
        self.airbot_play.switch_mode(RobotMode.PLANNING_POS)

    def _init_joystick(self):
        """Initialize joystick controller with callbacks"""
        self.joy = JoyController()
        self._setup_joy_callbacks()
        
    def _setup_joy_callbacks(self):
        """Setup joystick button and axis callbacks"""
        # Left stick X-axis (axis 1) for X movement
        self.joy.add_axis_callback(1, self._on_x_axis)
        
        # Right stick Y-axis (axis 3) for Y movement  
        self.joy.add_axis_callback(3, self._on_y_axis)
        
        # Left trigger (axis 2) for Z down
        self.joy.add_axis_callback(5, self._on_z_down)
        
        # Right trigger (axis 5) for Z up  
        self.joy.add_axis_callback(2, self._on_z_up)
        
        # Start button (7) to quit
        self.joy.add_button_down_callback(7, self._quit)
        
        # A button (4) to reset position
        self.joy.add_button_down_callback(0, self._start_collect)
        self.joy.add_button_down_callback(1, self._reset_position)


    def _start_collect(self):
        self.start_collect = True
        print("Start collecting data")
        
    def _on_x_axis(self, value):
        """Handle X-axis movement - store continuous value"""
        if self.pause:
            return
        # if abs(value) > self.deadzone:
        #     normalized_value = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        #     if value < 0:
        #         normalized_value = -normalized_value
        #     self.continuous_move_x = -normalized_value
        # else:
        #     self.continuous_move_x = 0
        self.continuous_move_x = -np.sign(value) * (value**2)  # Square for finer control
            
    def _on_y_axis(self, value):
        """Handle Y-axis movement - store continuous value"""
        if self.pause:
            return
        # if abs(value) > self.deadzone:
        #     normalized_value = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        #     if value < 0:
        #         normalized_value = -normalized_value
        #     self.continuous_move_y = -normalized_value
        # else:
        #     self.continuous_move_y = 0
        self.continuous_move_y = -np.sign(value) * (value**2)  # Square for finer control
            
    def _on_z_down(self, value):
        """Handle Z down movement - store continuous value"""
        if self.pause:
            return
        # trigger_value = (value + 1) / 2
        # if trigger_value > self.deadzone:
        #     self.continuous_move_z = -trigger_value
        # elif self.continuous_move_z < 0:
        #     self.continuous_move_z = 0
        value = (value + 1) / 2
        self.continuous_move_z = - (value**2)  # Square for finer
            
    def _on_z_up(self, value):
        """Handle Z up movement - store continuous value"""
        if self.pause:
            return
        # trigger_value = (value + 1) / 2
        # if trigger_value > self.deadzone:
        #     self.continuous_move_z = trigger_value
        # elif self.continuous_move_z > 0:
        #     self.continuous_move_z = 0
        value = (value + 1) / 2
        self.continuous_move_z = value**2  # Square for finer
            
    def _move_position(self, axis, delta):
        """Move position along specified axis and update robot target"""
        self.current_position[axis] += delta
        
        # Safety limits
        self.current_position[0] = np.clip(self.current_position[0], 0.1, 0.4)  # X limits
        self.current_position[1] = np.clip(self.current_position[1], -0.3, 0.3)  # Y limits  
        self.current_position[2] = np.clip(self.current_position[2], 0.095, 0.4)  # Z limits
        
        self._update_robot_target()
        
    def _update_continuous_movement(self, dt):
        """Update position based on continuous joystick input values"""
        if abs(self.continuous_move_x) > 0 or abs(self.continuous_move_y) > 0 or abs(self.continuous_move_z) > 0:
            # Apply movement with frame time
            dx = self.continuous_move_x * self.position_step * dt * 60  # Scale by 60 for smoother movement
            dy = self.continuous_move_y * self.position_step * dt * 60
            dz = self.continuous_move_z * self.position_step * dt * 60 * 2
            
            old_position = self.current_position.copy()
            
            self.current_position[0] += dx
            self.current_position[1] += dy
            self.current_position[2] += dz
            
            # Safety limits
            self.current_position[0] = np.clip(self.current_position[0], 0.1, 0.4)  # X limits
            self.current_position[1] = np.clip(self.current_position[1], -0.3, 0.3)  # Y limits  
            self.current_position[2] = np.clip(self.current_position[2], 0.095, 0.4)  # Z limits
            
            # Only update robot target if position actually changed
            if not np.allclose(old_position, self.current_position, atol=1e-6):
                self._update_robot_target()
        
    def _update_robot_target(self):
        """Solve IK and send target to robot"""
        try:
            # Solve inverse kinematics
            self.target_joints, converged = self.ik_solver.solve_ik(
                target_pos=self.current_position,
                target_ori=self.fixed_orientation,
                current_qpos=self.current_qpos
            )
            
            if converged:
                self.impedance_controller.set_target(self.target_joints)
                # # Send joint target to robot via AirbotPlay
                # self.airbot_play.set_target_joint_q(
                #     target_joints.tolist(), 
                #     blocking=False, 
                #     vel=0.1, 
                #     acceleration=0.1, 
                #     use_planning=True
                # )
                print(f"Position: [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}]")
            else:
                print("IK failed to converge")
                
        except Exception as e:
            print(f"Error updating robot target: {e}")
            
    def _reset_position(self):
        """Reset to starting position"""
        # Fixed orientation (pointing downward)
        self.pause = True
        time.sleep(0.1)
        self.airbot_play.switch_mode(RobotMode.PLANNING_POS)

        self.fixed_orientation = np.array([0.707, 0.0, 0.707, 0.0])  # qw, qx, qy, qz
        
        # Current end-effector position
        self.current_position = np.array([0.259, -0.026, 0.176])  # Starting position

        target_joints, _ = self.ik_solver.solve_ik(
            target_pos = self.current_position,
            target_ori = self.fixed_orientation,
            current_qpos = [0,0,0,0,0,0]
        )
        self._update_robot_target()
        self.airbot_play.move_to_joint_pos(list(target_joints), True)
        time.sleep(0.1)
        self.airbot_play.switch_mode(RobotMode.MIT_INTEGRATED)
        self.pause = False

        print("Position reset")
        
    def _quit(self):
        """Quit the application"""
        print("Shutting down...")
        self.running = False
        
    def _robot_state_loop(self):
        """Continuous loop to update robot state and send joint commands"""
        while self.running:
            if self.pause:
                time.sleep(0.004)
                continue
            try:
                # Get actual robot state from airbot_play
                self.current_qpos = np.array(self.airbot_play.get_joint_pos())
                self.current_qvel = np.array(self.airbot_play.get_joint_vel())
                self.current_tau = np.array(self.airbot_play.get_joint_eff())
                
                # print("position error: ", self.current_qpos-self.target_joints)
                
                self.impedance_controller.update_state(self.current_qpos, self.current_qvel, self.current_tau)
                self.current_ext_force = self.impedance_controller.get_ext_force()


                tau = self.impedance_controller.compute_torque()
                # too_large = False
                for i in range(len(tau)):
                    tau[i] = tau[i] * self.coeff[i]
                    # if tau[i] < -self.max_tau or tau[i] > self.max_tau:
                    #     too_large = True
                        
                # if too_large:
                #     print("Warning: Torque too large")
                #     continue
                tau = np.clip(tau, -self.max_tau, self.max_tau)
                self.airbot_play.mit_joint_integrated_control(list(self.target_joints),[0,0,0,0,0,0], list(tau), list(self.kp), list(self.kd))
                # print("target_tau",tau)
                # print("current_tau", self.current_tau)
                # calculate_coeff = []
                # for i in range(len(tau)):
                #     calculate_coeff.append((self.current_tau[i])/(tau[i]+1e-5))
                    
                # print("calculate_coeff", calculate_coeff)
                # print("ext force:", self.impedance_controller.get_ext_force())
                if not self.got_init_force:
                    self.got_init_force = True
                    self.init_force = self.current_ext_force
                # print("External force:", self.current_ext_force)
                delta_force = np.linalg.norm(self.current_ext_force-self.init_force)
                # print(f"Delta force: {delta_force:.3f} N")
                self.update_haptic_feedback(delta_force)

                time.sleep(0.004)  # 250Hz loop

            except Exception as e:
                print(f"Error in robot state loop: {e}")
                time.sleep(0.004)
                
    def start(self):
        """Start the robot control system"""
        print("Starting Robot Joy Controller")
        print(f"Controller: {self.joy.get_info()['name']}")
        print(f"Starting position: {self.current_position}")
        print(f"Fixed orientation (qw,qx,qy,qz): {self.fixed_orientation}")
        print("\nControls:")
        print("- Left stick X-axis: Move in X direction (left/right)")
        print("- Right stick Y-axis: Move in Y direction (forward/back)")
        print("- LT: Move down (Z-)")
        print("- RT: Move up (Z+)")
        print("- A: Reset position")
        print("- START: Quit")
        print("-" * 50)
        
        # Initialize robot to starting position
        # self.airbot_play.move_to_cart_pose([self.current_position, self.fixed_orientation], blocking=True)
        target_joints, _ = self.ik_solver.solve_ik(
            target_pos = self.current_position,
            target_ori = self.fixed_orientation,
            current_qpos = [0,0,0,0,0,0]
        )
        self.airbot_play.move_to_joint_pos(list(target_joints), True)
        self.airbot_play.switch_mode(RobotMode.MIT_INTEGRATED)
        # Initialize robot target
        self._update_robot_target()
        
        # Start robot state monitoring thread
        self.running = True
        self.state_thread = threading.Thread(target=self._robot_state_loop, daemon=True)
        self.state_thread.start()
        
        
        # Main control loop
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
            self._cleanup()
            
    def update_haptic_feedback(self, delta_force):
        """Update haptic feedback based on force magnitude"""
        # Define thresholds and corresponding rumble patterns
        if delta_force < 6.0:
            # No rumble
            return
        elif delta_force < 8.0:
            # Light rumble
            low_freq = 0
            high_freq = 0.3
            duration = 100
        elif delta_force < 10.0:
            # Medium rumble
            low_freq = 0
            high_freq = 0.6
            duration = 100
        else:
            # Strong rumble
            low_freq = 0.3
            high_freq = 1.0
            duration = 200
        
        self.joy.rumble(low_freq, high_freq, duration)
    def _cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if self.state_thread and self.state_thread.is_alive():
            self.state_thread.join(timeout=1.0)
        
        # self.airbot_play.switch_mode(RobotMode.PLANNING_POS)
        # self.airbot_play.move_to_joint_pos([0, 0, 0, 0, 0, 0], blocking=True)
        self.joy.quit()
        print("Cleanup complete")

if __name__ == "__main__":
    try:
        controller = RobotJoyController()
        controller.start()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. An Xbox controller connected")
        print("2. The robot properly connected and configured")