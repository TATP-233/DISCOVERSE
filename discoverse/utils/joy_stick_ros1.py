import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Joy

    class JoyTeleopRos1:
        def __init__(self) -> None:

            self.joy_cmd = Joy()
            self.joy_cmd.header.stamp = rospy.Time.now()
            self.joy_cmd.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            self.joy_cmd.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.last_buttons = np.zeros(11, np.bool_)
            self.raising_sig = np.zeros(11, np.bool_)
            self.falling_sig = np.zeros(11, np.bool_)
            self.joyCmdRecv = False

            rospy.Subscriber("/joy", Joy, self.joy_callback)

        def reset(self):
            self.joy_cmd.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            self.joy_cmd.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.raising_sig[:] = False
            self.falling_sig[:] = False
            self.joyCmdRecv = False

        def get_raising_edge(self, i):
            if i < len(self.raising_sig):
                if self.raising_sig[i]:
                    self.raising_sig[i] = False
                    return True
                else:
                    return False
            else:
                return None

        def get_falling_edge(self, i):
            if i < len(self.falling_sig):
                if self.falling_sig[i]:
                    self.falling_sig[i] = False
                    return True
                else:
                    return False
            else:
                return None

        def joy_callback(self, msg: Joy):
            self.joy_cmd = msg
            self.raising_sig = self.raising_sig | (
                np.array(msg.buttons) & ~self.last_buttons
            )
            self.falling_sig = self.falling_sig | (
                ~np.array(msg.buttons) & self.last_buttons
            )
            self.last_buttons = np.array(msg.buttons)
            self.joyCmdRecv = True

except ImportError:
    raise ImportError("ROS1: rospy is not imported. JoyTeleop is not available.")
