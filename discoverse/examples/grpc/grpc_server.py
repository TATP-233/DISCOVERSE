import grpc
from concurrent import futures
import numpy as np
import array_pb2
import array_pb2_grpc


class ArrayServiceServicer(array_pb2_grpc.ArrayServiceServicer):
    def __init__(self, n_obs=7, n_act=7):
        self.obs_array = np.zeros(n_obs, dtype=np.float32)
        self.action_array = np.zeros(n_act, dtype=np.float32)

    def GetArray(self, request, context):
        response = array_pb2.GetArrayResponse(data=self.obs_array.tolist())
        return response

    def SetArray(self, request, context):
        try:
            self.action_array = np.array(request.data, dtype=np.float32)
            success = True
        except Exception as e:
            print(f"Error setting array: {e}")
            success = False

        response = array_pb2.SetArrayResponse(success=success)
        return response


def sim_serve(servicer, nw=5, blocking=False):
    """
    启动一个gRPC服务器，用于处理数组服务请求。

    参数:
    servicer (ArrayServiceServicer): 用于处理数组服务请求的服务实例。
    nw (int, optional): 用于处理请求的工作线程数。默认为5。
    blocking (bool, optional): 是否阻塞当前线程，直到服务器终止。默认为False。
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=nw))
    array_pb2_grpc.add_ArrayServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    if blocking:
        server.wait_for_termination()
    return server


if __name__ == "__main__":
    n_obs, n_act = 7, 7
    servicer = ArrayServiceServicer(n_obs, n_act)
    sim_serve(servicer, blocking=True)
