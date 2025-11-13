"""
The class for cooperative aerial-ground control (SkyRover & AIRBOT: cooperative_aerial_ground_sim.xml).

Author: Weibin Gu, 8-Nov-2024.
"""
import os
import time
import numpy as np
from pid_control import PIDControl as SkyroverPIDControl
from quadrotor_logger import QuadrotorLogger
import control_util as cu
from rm2_control import PIDControl as Rm2PIDControl
from rm2_control import compute_diff_drive_control

class CooperativeAerialGroundControl():
    def __init__(self, multi_robot):
        self.multi_robot = multi_robot
        self.dt = self.multi_robot.mj_model.opt.timestep # sampling time [sec]
        
        """ Differential drive control """
        self.diffd_lin_ctrl = Rm2PIDControl(kp=5.0, ki=0.1, kd=5)
        self.diffd_ang_ctrl = Rm2PIDControl(kp=1.0, ki=0.0, kd=0.0)
        
        """ Flight control and data logging """
        self.flight_ctrl = SkyroverPIDControl()
        self.logger = QuadrotorLogger()
        self.flight_ctrl_2 = SkyroverPIDControl()
        self.flight_ctrl_3 = SkyroverPIDControl()
        
        """ SkyRover waypoints """
        alt_target = 1.5
        self.waypoints = {
            "phase_1": [
                [0.0, 0.0, alt_target],
                [-2.0, 0.0, alt_target],
                # [-2.0, 0.0, 0.8*alt_target],
                # [-2.0, 0.0, 0.6*alt_target],
                # [-2.0, 0.0, 0.4*alt_target],
                # [-2.0, 0.0, 0.2*alt_target],
                [-2.0, -0.0, 0.1]
            ],
            "phase_2": [
                [-1.0, 2.0, 0.0]
                ],
            "phase_3": [
            ]
        }
        self.waypoint_tolerance = 0.3 #0.18
        self.current_waypoint_index = 0
        self.target_pos = [0., 0., 0.] # dummy (for logging)

        """ Task scheduler flag """
        self.mission_phase = 0
        self.mission_start_time = time.time()
        self.last_print_time = time.time()
        self.mission_completed = False
        self.sine_motion_start_time = time.time()

    def do_mission(self, mission_phase):
        if mission_phase != 0:
            t = time.time() - self.sine_motion_start_time
            sine_motion = 5*np.sin(0.1*t) - 3
            self.multi_robot.mj_data.ctrl[23] = sine_motion
            self.multi_robot.mj_data.ctrl[24] = sine_motion
            self.multi_robot.mj_data.ctrl[25] = sine_motion
            self.multi_robot.mj_data.ctrl[26] = sine_motion
            self.multi_robot.mj_data.ctrl[27] = sine_motion
            self.multi_robot.mj_data.ctrl[28] = sine_motion
            self.multi_robot.mj_data.ctrl[29] = sine_motion
            self.multi_robot.mj_data.ctrl[30] = sine_motion

        # self.execute_skyrover_fly_background(target_pos=[10, 10, 6], robot="skyrover_1")
        # self.execute_skyrover_fly(target_pos=[10, -10, 8], robot="skyrover_2")

        """ Phase 0: Initialization and Landing
            The SkyRover and RM2 are initialized and spawned in the MuJoCo environment. SkyRover, in drone mode, lands on top of RM2. """
        if mission_phase == 0:
            # Wait for 5 seconds to start the simulation
            wait_time = 5
            remaining_time = int(wait_time - (time.time() - self.mission_start_time))
            if time.time() - self.last_print_time >= 1:
                if remaining_time > 0:
                    print(f"Countdown: {remaining_time} seconds remaining to start...")
                self.last_print_time = time.time()
            if time.time() - self.mission_start_time > 5:
                self.mission_phase = self.mission_phase + 1
                print(f"Simulation started. Executing mission phase #{self.mission_phase}...")
                wp_completed = False # initialization for next mission

        """ Phase 1: Waypoint Navigation and Landing
            SkyRover takes off, navigates to a specified waypoint of interest, and performs a landing at the target location. """
        if mission_phase == 1:
            pos, _, _, _, _ = self.multi_robot.get_skyrover_state()
            wp_completed, target_pos = self.waypoint_checker(pos=pos, phase="phase_1")
            if not wp_completed:
                self.execute_skyrover_fly(target_pos=target_pos)
                self.target_pos = target_pos
                dummy_rpm = 1.0
                self.multi_robot.mj_data.ctrl[4] = dummy_rpm
                self.multi_robot.mj_data.ctrl[5] = dummy_rpm
                self.multi_robot.mj_data.ctrl[9] = dummy_rpm
                self.multi_robot.mj_data.ctrl[10] = dummy_rpm    
            else:
                self.execute_skyrover_fly(land_flag=True)
                dummy_rpm = 0.0
                self.multi_robot.mj_data.ctrl[4] = dummy_rpm
                self.multi_robot.mj_data.ctrl[5] = dummy_rpm
                self.multi_robot.mj_data.ctrl[9] = dummy_rpm
                self.multi_robot.mj_data.ctrl[10] = dummy_rpm
                self.mission_phase = self.mission_phase + 1
                print(f"Executing mission phase #{self.mission_phase}...")                

        """ Phase 2: Rover Mode Transition and Movement
            SkyRover transitions to rover mode and drives autonomously for a set duration. """
        if mission_phase == 2:
            if self.mission_completed == False:
                self.mission_completed = self.drone_to_rover()
                self.mission_start_time = time.time() # initialization for next mission
            else: 
                drive_time = 2
                elapsed_time = time.time() - self.mission_start_time
                if elapsed_time < drive_time:
                    self.execute_skyrover_drive([-0.35, -0.35, 0.15, 0.15])
                else:
                    self.execute_skyrover_drive([0, 0, 0, 0])
                    self.mission_phase = self.mission_phase + 1
                    print(f"Executing mission phase #{self.mission_phase}...")
                    self.mission_completed = False # initialization for next mission
                    wp_completed = False # initialization for next mission
                    self.current_waypoint_index = 0 # initialization for next mission

        """ Phase 3: Return to RM2 and Landing
            SkyRover reverts to drone mode, takes off, and returns to RM2's location. RM2 advances slowly while SkyRover aligns and lands on top of RM2. """
        if mission_phase == 3:
            alt_target = 12.0
            if self.mission_completed == False:
                self.mission_completed = self.rover_to_drone()
                pos, _, _, _, _ = self.multi_robot.get_skyrover_state()
                self.waypoints["phase_3"] = [
                    [pos[0], pos[1], alt_target] 
                ]
            else:
                """ Get states """
                skyrover_pos, _, _, _, _ = self.multi_robot.get_skyrover_state()
                rm2_pos, _, _, _, _ = self.multi_robot.get_rm2_state()

                """ Target """
                rm2_waypoint = [3.0, 0.0]
                landing_displacement = 0.12

                """ Update waypoint list """
                temp_x_1 = self.waypoints["phase_3"][0][0]
                temp_y_1 = self.waypoints["phase_3"][0][1] 
                temp_x_2 = self.waypoints["phase_3"][0][0]
                temp_y_2 = 0.0
                temp_x_3 = self.waypoints["phase_3"][0][0] 
                temp_y_3 = 0.0
                temp_x_4 = 0.0
                temp_y_4 = 0.0
                temp_x_5 = rm2_waypoint[0]
                temp_y_5 = 0.0
                temp_x_6 = rm2_waypoint[0] - landing_displacement
                temp_y_6 = 0.0
                temp_x_7 = rm2_waypoint[0] - landing_displacement
                temp_y_7 = 0.0
                if len(self.waypoints["phase_3"]) == 1:
                    self.waypoints["phase_3"].append([temp_x_1, 0.8*temp_y_1, alt_target])
                    self.waypoints["phase_3"].append([temp_x_1, 0.6*temp_y_1, alt_target])
                    self.waypoints["phase_3"].append([temp_x_1, 0.4*temp_y_1, alt_target])
                    self.waypoints["phase_3"].append([temp_x_1, 0.2*temp_y_1, alt_target])
                    self.waypoints["phase_3"].append([temp_x_2, temp_y_2, alt_target])
                    for i in range(100):
                        self.waypoints["phase_3"].append([(1-i/100)*temp_x_3, temp_y_3, alt_target])
                    self.waypoints["phase_3"].append([temp_x_4, temp_y_4, alt_target])
                    for i in range(100):
                        self.waypoints["phase_3"].append([i/100*temp_x_5, temp_y_5, alt_target]) 
                    self.waypoints["phase_3"].append([temp_x_5, temp_y_5, alt_target]) 
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, alt_target])
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, 0.8*alt_target])
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, 0.6*alt_target])
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, 0.4*alt_target])
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, 0.2*alt_target])
                    self.waypoints["phase_3"].append([temp_x_6, temp_y_6, 0.1*alt_target])
                    self.waypoints["phase_3"].append([temp_x_7, temp_y_7, 1.0])

                """ Differential drive control """
                self.execute_rm2(target_pos=rm2_waypoint)

                """ Flight control """
                dummy_rpm = 1.0
                self.multi_robot.mj_data.ctrl[4] = dummy_rpm
                self.multi_robot.mj_data.ctrl[5] = dummy_rpm
                self.multi_robot.mj_data.ctrl[9] = dummy_rpm
                self.multi_robot.mj_data.ctrl[10] = dummy_rpm 
                self.flight_ctrl.D_COEFF_TOR = np.array([20000., 20000., -2880.])
                wp_completed, target_pos = self.waypoint_checker(pos=skyrover_pos, phase="phase_3")
                # print(skyrover_pos, target_pos)
                # print(rm2_pos,rm2_waypoint)
                if not wp_completed:
                    self.execute_skyrover_fly(target_pos=target_pos)
                    self.target_pos = target_pos
                else:
                    self.execute_skyrover_fly(land_flag=True)
                    dummy_rpm = 0.0
                    self.multi_robot.mj_data.ctrl[4] = dummy_rpm
                    self.multi_robot.mj_data.ctrl[5] = dummy_rpm
                    self.multi_robot.mj_data.ctrl[9] = dummy_rpm
                    self.multi_robot.mj_data.ctrl[10] = dummy_rpm 
                    self.mission_phase = self.mission_phase + 1
                    print(f"Executing mission phase #{self.mission_phase}...")       
                    self.mission_start_time = time.time() # initialization for next mission

        """ Mission completed """           
        if mission_phase > 3:
            wait_time = 3
            if time.time() - self.mission_start_time > wait_time:
                print(f"Mission completed.")
                return True
        
        return False

    def waypoint_checker(self, pos, phase):
        target_pos = self.waypoints[phase][self.current_waypoint_index]
        # Calculate position error to see if we are within tolerance of the waypoint
        pos_error = np.linalg.norm(np.array(target_pos) - np.array(pos))
        if pos_error < self.waypoint_tolerance:
            print(f"Reached waypoint {self.current_waypoint_index + 1}")
            self.current_waypoint_index += 1  # Move to the next waypoint
            if self.current_waypoint_index >= len(self.waypoints[phase]):
                print("All waypoints reached. Exiting...")
                return True, []
            target_pos = self.waypoints[phase][self.current_waypoint_index]
        return False, target_pos
    
    def execute_rm2(self, target_pos):
        pos, vel, quat, pqr, eul = self.multi_robot.get_rm2_state()
        control_input = compute_diff_drive_control(pid_linear=self.diffd_lin_ctrl,
                                                   pid_angular=self.diffd_ang_ctrl,
                                                   robot_position=pos[:2],
                                                   robot_orientation=eul[-1],
                                                   target_position=target_pos,
                                                   wheel_radius=self.multi_robot.rm2_wheel_radius,
                                                   robot_radius=self.multi_robot.rm2_robot_radius,
                                                   dt=self.dt)
        self.multi_robot.rm2_drive(control_input)

    def execute_skyrover_drive(self, diff_drive_ctrl):
        self.multi_robot.skyrover_drive(diff_drive_ctrl)
    
    def execute_skyrover_fly(self, 
                          target_pos=[0., 0., 0.], 
                          target_eul=[0., 0., 0.], 
                          target_vel=[0., 0., 0.], 
                          target_pqr=[0., 0., 0.],
                          land_flag=False,
                          robot="skyrover"
                          ):
        pos, vel, quat, pqr, eul = self.multi_robot.get_skyrover_state(robot=robot)
        state = np.concatenate((pos, quat, vel, pqr))
        rpm, err_pos, err_rpy = self.flight_ctrl.compute_control_from_state(control_timestep=self.dt,
                                                                state=state,
                                                                target_pos=target_pos,
                                                                target_rpy=target_eul,
                                                                target_vel=target_vel,
                                                                target_rpy_rates=target_pqr
                                                                )
        scaled_rpm = cu.scale_rpm_array(rpm) # to be compatible with Mujoco actuators
        # print("Controller rpm: ", rpm)
        # print("Scaled rpm: ", scaled_rpm)
        # print("Position error: ", err_pos)
        # print("Position: ", pos)
        if land_flag:
            self.multi_robot.skyrover_fly([0, 0, 0, 0], robot)
        else:
            self.multi_robot.skyrover_fly(scaled_rpm, robot)

    def execute_skyrover_fly_background(self, 
                          target_pos=[0., 0., 0.], 
                          target_eul=[0., 0., 0.], 
                          target_vel=[0., 0., 0.], 
                          target_pqr=[0., 0., 0.],
                          land_flag=False,
                          robot="skyrover"
                          ):
        pos, vel, quat, pqr, eul = self.multi_robot.get_skyrover_state(robot=robot)
        state = np.concatenate((pos, quat, vel, pqr))
        rpm, err_pos, err_rpy = self.flight_ctrl_2.compute_control_from_state(control_timestep=self.dt,
                                                                state=state,
                                                                target_pos=target_pos,
                                                                target_rpy=target_eul,
                                                                target_vel=target_vel,
                                                                target_rpy_rates=target_pqr
                                                                )
        scaled_rpm = cu.scale_rpm_array(rpm) # to be compatible with Mujoco actuators
        print("Controller rpm: ", rpm)
        print("Scaled rpm: ", scaled_rpm)
        print("Position error: ", err_pos)
        print("Position: ", pos)

        if robot == "skyrover_1":
            idx = [31, 32, 33, 34]
        if robot == "skyrover_2":
            idx = [35, 36, 37, 38]

        if land_flag:
            self.multi_robot.mj_data.ctrl[idx[0]] = 0
            self.multi_robot.mj_data.ctrl[idx[1]] = 0
            self.multi_robot.mj_data.ctrl[idx[2]] = 0
            self.multi_robot.mj_data.ctrl[idx[3]] = 0
        else:
            self.multi_robot.mj_data.ctrl[idx[0]] = scaled_rpm[0]
            self.multi_robot.mj_data.ctrl[idx[1]] = scaled_rpm[1]
            self.multi_robot.mj_data.ctrl[idx[2]] = scaled_rpm[2]
            self.multi_robot.mj_data.ctrl[idx[3]] = scaled_rpm[3]

    def rover_to_drone(self):
        while self.multi_robot.locomode != "drone":
            _ = self.multi_robot.switch_locomotion(locomode="drone")
            return False
        return True
    
    def drone_to_rover(self):
        while self.multi_robot.locomode != "rover":
            _ = self.multi_robot.switch_locomotion(locomode="rover")
            return False
        return True
    
    def _get_all_robot_state(self):
        rm2_pos, rm2_vel, rm2_quat, rm2_pqr, rm2_eul = self.multi_robot.get_rm2_state(verbose=False)
        skyrover_pos, skyrover_vel, skyrover_quat, skyrover_pqr, skyrover_eul = self.multi_robot.get_skyrover_state(verbose=False)
        skyrover_state = np.concatenate((skyrover_pos, skyrover_quat, skyrover_vel, skyrover_pqr))
        return skyrover_state, rm2_pos

