"""This is a Sample Python file with module-level docstring."""

from __future__ import annotations

import math
import struct
import time
from threading import Thread

# pip install python-can
import can

class AirbotEncoder:
    """
    AIRBOT Encoder class to operate encoder
    """

    def __init__(self,
                 can_channel: str = "can0",
                 id: int = 1,
                 auto_control: bool = False,
                 control_period: float = 0.01,
                 normalize: bool = False,
                 limits: tuple = (0, 2 * math.pi)):
        """
        Initialize the AIRBOT Encoder class

        Params:
        can_channel (str): The CAN channel to use for communication
        id (int): The ID of the encoder
        auto_control (bool): Whether to run the control loop
        control_period (float): The control period
        """
        self.can_channel = can_channel
        self.bus = can.interface.Bus(channel=self.can_channel,
                                     bustype="socketcan")
        self.id = id
        self.pos = 0.
        self.normalize = normalize
        self.limits = limits
        self.thread = Thread(target=self.receive_messages)
        self.thread.start()
        if auto_control:
            self.control_thread = Thread(target=self.control_loop,
                                         args=(control_period,))
            self.control_thread.start()

    def receive_messages(self):
        """
        Receive messages from the encoder
        """
        for msg in self.bus:
            if msg.arbitration_id == self.id | 0x100:
                # print(msg)
                if msg.data[0] == 0x11:
                    pos_ptr = msg.data[2:6]
                    pos = struct.unpack('<f',
                                        bytes(pos_ptr))[0] / 360.0 * 2 * math.pi
                    if self.normalize:
                        pos = min(
                            1,
                            max(0, (pos - self.limits[0]) /
                                (self.limits[1] - self.limits[0])))
                    self.pos = pos

    def control_loop(self, control_period: float):
        """
        Control loop for the encoder

        Params:
        control_period (float): The control period
        """
        while True:
            self.read_request()
            time.sleep(control_period)

    def read_request(self):
        """
        Send a read request to the encoder
        """
        msg = can.Message(arbitration_id=self.id,
                          data=[0x11],
                          is_extended_id=False)
        self.bus.send(msg)


class AirbotReplay:
    """
    AIRBOT Replay class
    """

    def __init__(self,
                 can_channel: str = "can0",
                 with_eef: bool = True,
                 auto_control: bool = False,
                 control_period: float = 0.01):
        """
        Initialize the AIRBOT Replay class

        Params:
        can_channel (str): The CAN channel to use for communication
        with_eef (bool): Whether the end effector is present
        auto_control (bool): Whether to run the control loop
        control_period (float): The control period
        """
        self.can_channel = can_channel
        self.bus = can.interface.Bus(channel=self.can_channel,
                                     bustype="socketcan")
        self.with_eef = with_eef
        self.encoders = [
            AirbotEncoder(can_channel, i, auto_control, control_period)
            for i in range(1, 7)
        ]
        if with_eef:
            self.encoders.append(
                AirbotEncoder(can_channel,
                              7,
                              auto_control,
                              control_period,
                              normalize=True,
                              limits=(0, -2.76)))

    def read_request(self):
        """
        Send a read request to the encoders
        """
        for encoder in self.encoders:
            encoder.read_request()


def main():
    """
    Main function
    """
    replay = AirbotReplay("can0",
                          with_eef=True,
                          auto_control=True,
                          control_period=0.1)
    for _ in range(100):
        time.sleep(0.1)
        print([encoder.pos for encoder in replay.encoders])


if __name__ == "__main__":
    main()
