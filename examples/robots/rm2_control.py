"""
A class of PID controller for differential drive robots.

Note: This is not the correct code for differential drive PID control. A workaround is used to set the sign of the linear velocity as follows
    if position_error[0] > 0:  # Robot is behind the target in x direction
        linear_velocity = -linear_velocity"
This is, however, wrong, and will be fixed in future updates.

TODO: Fix bugs.

Author: Weibin Gu, 9-Nov-2024.
"""
import matplotlib.pyplot as plt
import numpy as np

# PID Controller Class
class PIDControl():
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0  # Previous error value
        self.integral = 0  # Integral of error

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def compute_diff_drive_control(pid_linear,
                               pid_angular,
                               robot_position, 
                               robot_orientation, 
                               target_position, 
                               wheel_radius,
                               robot_radius, 
                               dt):
    # Compute the error in position
    position_error = target_position - robot_position
    distance_error = np.linalg.norm(position_error)

    # Compute the error in orientation
    target_angle = np.arctan2(position_error[1], position_error[0])
    orientation_error = target_angle - robot_orientation

    # Normalize the angle error to be between -pi and pi
    orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))

    # Compute linear and angular velocities using PID controllers
    linear_velocity = pid_linear.update(distance_error, dt) 
    angular_velocity = pid_angular.update(orientation_error, dt)

    # The sign of the linear velocity should depend on the direction of the position error
    # If the position error is positive (robot needs to go forward), the velocity should be positive
    # If negative (robot needs to go backward), the velocity should be negative
    if position_error[0] > 0:  # Robot is behind the target in x direction
        linear_velocity = -linear_velocity

    # Compute control inputs based on differential drive kinematics
    # Linear and angular velocities to wheel velocities
    v_left = (linear_velocity - angular_velocity * robot_radius) / wheel_radius
    v_right = (linear_velocity + angular_velocity * robot_radius) / wheel_radius

    # Compute the wheel speeds (4 wheels for a differential drive)
    front_left = v_left
    front_right = v_right
    back_left = v_left
    back_right = v_right

    # Store control inputs 
    control_input = [front_left, front_right, back_right, back_left]

    return control_input