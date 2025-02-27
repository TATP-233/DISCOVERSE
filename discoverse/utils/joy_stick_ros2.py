import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy

    class JoyTeleopRos2(Node):
        def __init__(self, node_name='joy_teleop_node') -> None:
            super().__init__(node_name)

            self.joy_cmd = Joy()
            self.joy_cmd.header.stamp = self.get_clock().now().to_msg()
            # 确保axes数组有6个元素
            self.joy_cmd.axes = [0.0] * 6
            # 确保buttons数组有12个元素
            self.joy_cmd.buttons = [0] * 12
            self.last_buttons = np.zeros(12, np.bool_)
            self.raising_sig = np.zeros(12, np.bool_)
            self.falling_sig = np.zeros(12, np.bool_)
            self.joyCmdRecv = False

            self.subscription = self.create_subscription(
                Joy,
                '/joy',
                self.joy_callback,
                10)

        def reset(self):
            # 确保axes数组有6个元素
            self.joy_cmd.axes = [0.0] * 6
            # 确保buttons数组有12个元素
            self.joy_cmd.buttons = [0] * 12
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
            # 保存原始消息前确保数组长度正确
            
            # 确保axes数组长度正确
            axes_array = np.array(msg.axes)
            expected_axes_length = 6  # 期望的axes数组长度
            
            if len(axes_array) != expected_axes_length:
                print(f"警告：axes数量不匹配。收到 {len(axes_array)}，预期 {expected_axes_length}")
                # 调整axes数组大小
                if len(axes_array) > expected_axes_length:
                    # 如果收到的axes数量多于预期，截取
                    msg.axes = list(axes_array[:expected_axes_length])
                else:
                    # 如果收到的axes数量少于预期，用0填充
                    temp = [0.0] * expected_axes_length
                    for i in range(len(axes_array)):
                        temp[i] = axes_array[i]
                    msg.axes = temp
            
            # 保存调整后的消息
            self.joy_cmd = msg
            
            # 确保buttons数组长度正确
            buttons_array = np.array(msg.buttons)
            if len(buttons_array) != len(self.last_buttons):
                print(f"警告：按钮数量不匹配。收到 {len(buttons_array)}，预期 {len(self.last_buttons)}")
                # 如果收到的按钮数量与预期不同，调整数组大小
                if len(buttons_array) > len(self.last_buttons):
                    buttons_array = buttons_array[:len(self.last_buttons)]
                else:
                    # 如果收到的按钮数量少于预期，用0填充
                    temp = np.zeros(len(self.last_buttons), np.bool_)
                    temp[:len(buttons_array)] = buttons_array
                    buttons_array = temp
            
            self.raising_sig = self.raising_sig | (buttons_array & ~self.last_buttons)
            self.falling_sig = self.falling_sig | (~buttons_array & self.last_buttons)
            self.last_buttons = buttons_array
            self.joyCmdRecv = True

except ImportError:
    # 当ROS2不可用时，提供一个模拟类
    print("警告: ROS2 (rclpy) 未安装。使用模拟的JoyTeleopRos2类。")
    
    class Joy:
        def __init__(self):
            self.header = type('obj', (object,), {
                'stamp': None
            })
            # 根据实际情况，设置为6个元素
            self.axes = [0.0] * 6
            # 确保buttons数组有12个元素
            self.buttons = [0] * 12
    
    class JoyTeleopRos2:
        def __init__(self, node_name='joy_teleop_node') -> None:
            self.joy_cmd = Joy()
            self.joy_cmd.header.stamp = None
            self.last_buttons = np.zeros(12, np.bool_)
            self.raising_sig = np.zeros(12, np.bool_)
            self.falling_sig = np.zeros(12, np.bool_)
            self.joyCmdRecv = False
            print("模拟JoyTeleopRos2已初始化。请注意，这不会接收真实的手柄输入。")

        def reset(self):
            # 确保axes数组有6个元素
            self.joy_cmd.axes = [0.0] * 6
            # 确保buttons数组有12个元素
            self.joy_cmd.buttons = [0] * 12
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
        
        def joy_callback(self, msg):
            # 保存原始消息前确保数组长度正确
            
            # 确保axes数组长度正确
            axes_array = np.array(msg.axes)
            expected_axes_length = 6  # 期望的axes数组长度
            
            if len(axes_array) != expected_axes_length:
                print(f"警告：axes数量不匹配。收到 {len(axes_array)}，预期 {expected_axes_length}")
                # 调整axes数组大小
                if len(axes_array) > expected_axes_length:
                    # 如果收到的axes数量多于预期，截取
                    msg.axes = list(axes_array[:expected_axes_length])
                else:
                    # 如果收到的axes数量少于预期，用0填充
                    temp = [0.0] * expected_axes_length
                    for i in range(len(axes_array)):
                        temp[i] = axes_array[i]
                    msg.axes = temp
            
            # 保存调整后的消息
            self.joy_cmd = msg
            
            # 确保buttons数组长度正确
            buttons_array = np.array(msg.buttons)
            if len(buttons_array) != len(self.last_buttons):
                print(f"警告：按钮数量不匹配。收到 {len(buttons_array)}，预期 {len(self.last_buttons)}")
                # 如果收到的按钮数量与预期不同，调整数组大小
                if len(buttons_array) > len(self.last_buttons):
                    buttons_array = buttons_array[:len(self.last_buttons)]
                else:
                    # 如果收到的按钮数量少于预期，用0填充
                    temp = np.zeros(len(self.last_buttons), np.bool_)
                    temp[:len(buttons_array)] = buttons_array
                    buttons_array = temp
            
            self.raising_sig = self.raising_sig | (buttons_array & ~self.last_buttons)
            self.falling_sig = self.falling_sig | (~buttons_array & self.last_buttons)
            self.last_buttons = buttons_array
            self.joyCmdRecv = True
                
        def destroy_node(self):
            pass
