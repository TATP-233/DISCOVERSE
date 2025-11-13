"""
A class of PID controller for quadrotor flight.
Reference: https://github.com/wgu93/gym_pybullet_drones/blob/master/control/DSLPIDControl.py

Author: Weibin Gu, 4-Nov-2024.
"""
import numpy as np
import math
from scipy.spatial.transform import Rotation
import control_util as cu

class PIDControl():
    def __init__(self) -> None:
        """ Manually tuned PID gains """
        self.P_COEFF_FOR = np.array([5.5, 5.5, 2.25])
        self.I_COEFF_FOR = np.array([.5, .5, .5])
        self.D_COEFF_FOR = np.array([6.2, 6.2, 4.5])
        self.P_COEFF_TOR = np.array([70000., 70000., -3000.])
        self.I_COEFF_TOR = np.array([5., 5., -1.])
        self.D_COEFF_TOR = np.array([20000., 20000., -360.])

        self.KF = 4.62e-8 # the coeff converting RPMs to thrust

        """ PyBullet settings """
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        """ SkyRover specs """
        self.mass = 1.9
        self.gravity_const = 9.8
        self.GRAVITY = self.mass * self.gravity_const

        # Motor mixer: motor_pwm = MIXER_MATRIX * [thrust, roll, pitch, yaw]'
        # (Reference: https://cookierobotics.com/066/)
        self.MIXER_MATRIX = np.array([ 
            [1, 1, -1, 1],
            [1, 1, 1, -1],
            [1, -1, 1, 1],
            [1, -1, -1, -1],
            ])
        
        self.reset()
        
    def compute_control_from_state(self,
                                   control_timestep,
                                   state,
                                   target_pos,
                                   target_rpy=np.zeros(3),
                                   target_vel=np.zeros(3),
                                   target_rpy_rates=np.zeros(3),
                                   target_acc=np.zeros(3)
                                   ):
        return self._compute_control(control_timestep,
                                     cur_pos=state[0:3],
                                     cur_quat=state[3:7],
                                     cur_vel=state[7:10],
                                     cur_ang_vel=state[10:13],
                                     target_pos=target_pos,
                                     target_rpy=target_rpy,
                                     target_vel=target_vel,
                                     target_rpy_rates=target_rpy_rates,
                                     target_acc=target_acc
                                     )
    
    def _compute_control(self,
                         control_timestep,
                         cur_pos,
                         cur_quat,
                         cur_vel,
                         cur_ang_vel,
                         target_pos,
                         target_rpy,
                         target_vel,
                         target_rpy_rates,
                         target_acc
                         ):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._pidPositionControl(control_timestep,
                                                                      cur_pos,
                                                                      cur_quat,
                                                                      cur_vel,
                                                                      target_pos,
                                                                      target_rpy,
                                                                      target_vel
                                                                      )
        rpm = self._pidAttitudeControl(control_timestep,
                                       thrust,
                                       cur_quat,
                                       computed_target_rpy,
                                       target_rpy_rates
                                       )
        cur_rpy = cu.get_euler_from_quaternion(cur_quat)
        # print("computed_target_rpy: ", np.rad2deg(computed_target_rpy))
        # print("cur_rpy: ", np.rad2deg(cur_rpy))
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
        
    def _pidPositionControl(self,
                            control_timestep,
                            cur_pos,
                            cur_quat,
                            cur_vel,
                            target_pos,
                            target_rpy,
                            target_vel
                            ):
        cur_rotation = np.array(cu.get_matrix_from_quaternion(cur_quat))
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # print("Target thrust: ", target_thrust, ", Scalar thrust: ", scalar_thrust, ", Thrust: ", thrust)
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        # print("Target body z: ", target_z_ax)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e

    def _pidAttitudeControl(self,
                            control_timestep,
                            thrust,
                            cur_quat,
                            target_euler,
                            target_rpy_rates
                            ):
        cur_rotation = np.array(cu.get_matrix_from_quaternion(cur_quat))
        cur_rpy = np.array(cu.get_euler_from_quaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        # print("Current rotation: ", cur_rotation, ", Target rotation: ", target_rotation)
        # print("Rotation error: ", rot_e)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX[:, 1:], target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        self.control_counter = 0
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
