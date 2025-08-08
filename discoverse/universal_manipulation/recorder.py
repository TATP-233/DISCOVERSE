import os
import json
import fractions
import numpy as np

import av
import av.video

class PyavImageEncoder:
    def __init__(self, width: int, height: int, save_path: str, id: str|int):
        self.width = width
        self.height = height
        self.av_file_path = os.path.join(save_path, f"cam_{id}.mp4")
        if os.path.exists(self.av_file_path):
            os.remove(self.av_file_path)
        container = av.open(self.av_file_path, "w", format="mp4")
        stream: av.video.stream.VideoStream = container.add_stream("h264", options={"preset": "fast"})
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        self._time_base = int(1e6)
        stream.time_base = fractions.Fraction(1, self._time_base)
        self.container = container
        self.stream = stream
        self.start_time = None
        self.last_time = None
        self._cnt = 0

    def encode(self, image: np.ndarray, timestamp: float):
        self._cnt += 1
        if self.start_time is None:
            self.start_time = timestamp
            self.last_time = 0
            self.container.metadata["comment"] = str({"base_stamp": int(self.start_time * self._time_base)})
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        cur_time = timestamp
        frame.pts = int((cur_time - self.start_time) * self._time_base)
        frame.time_base = self.stream.time_base
        assert cur_time > self.last_time, f"Time error: {cur_time} <= {self.last_time}"
        self.last_time = cur_time
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        self.container = None

    def remove_av_file(self):
        if os.path.exists(self.av_file_path):
            os.remove(self.av_file_path)
            print(f">>>>> Removed {self.av_file_path}")

def recoder_single_arm(save_path, obs_lst):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        save_dict = {
            "time" : [],
            "obs"  : {
                "jq" : []
            },
            "act"  : []
        }
        for obs in obs_lst:
            save_dict["time"].append(obs['time'])
            save_dict["obs"]["jq"].append(obs['jq'])
            save_dict["act"].append(obs['action'])
        json.dump(save_dict, fp)