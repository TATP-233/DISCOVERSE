import numpy as np
import yaml

def get_matrix_from_quaternion(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    
    Args:
    - quat: array-like, shape (4,)
    Quaternion in the form [w, x, y, z]
    
    Returns:
    - rotation_matrix: np.ndarray, shape (3, 3)
    The 3x3 rotation matrix.
    """
    # Extract quaternion components
    w, x, y, z = quat
    
    # Calculate elements of the rotation matrix
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    
    return rotation_matrix

def get_euler_from_quaternion(quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    
    Args:
    - quat: array-like, shape (4,)
      Quaternion as [w, x, y, z]
      
    Returns:
    - euler_angles: tuple (roll, pitch, yaw)
      Euler angles in radians
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def get_body_rate_from_quaternion(quat, quat_dot, dt):
    """
    Compute the body angular velocity (body rate) from quaternion and its derivative.
    (Reference: https://mariogc.com/post/angular-velocity-quaternions/)

    Args:
    - quat: array-like, shape (4,)
    Quaternion representing the orientation, in the form [w, x, y, z].
    - quat_dot: array-like, shape (4,)
    Time derivative of the quaternion.

    Returns:
    - omega: np.ndarray, shape (3,)
    The body angular velocity vector [wx, wy, wz].
    """
    q1 = quat
    q2 = quat + quat_dot * dt
    
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def vee_op(S):
    """ 
    Convert a skew-symmetric matrix S to a vector using the vee operator.

    Args:
        S (numpy.ndarray): A 3x3 skew-symmetric matrix.
    
    Returns:
        numpy.ndarray: A vector of shape (3,).
    """
    if S.shape != (3, 3):
        raise ValueError("[ERROR] Input must be a 3x3 skew-symmetric matrix.")
    return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

def hat_op(v):
    """ Convert a vector to a skew-symmetric matrix. """
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

def load_param_from_yaml(file_path):
    """ Load geometric control parameters from yaml file. """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    gain_pos = config["gain_pos"]
    gain_vel = config["gain_vel"]
    gain_rotation = config["gain_rotation"]
    gain_rpy_rate = config["gain_rpy_rate"]
    return gain_pos, gain_vel, gain_rotation, gain_rpy_rate
    
def scale_rpm_array(rpm_array, old_min=0, old_max=21666.4475, new_min=0, new_max=10):
    """ Scale each entry in the RPM array from 21666.4475 (PWM=65535) to 10 """
    return (rpm_array - old_min) * (new_max - new_min) / (old_max - old_min) + new_min