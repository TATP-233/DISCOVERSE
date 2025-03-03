import time
import grpc
import numpy as np
import array_pb2
import array_pb2_grpc


def run(nj):
    channel = grpc.insecure_channel("localhost:50051")
    stub = array_pb2_grpc.ArrayServiceStub(channel)

    set_response = False
    action_array = np.zeros(nj, dtype=np.float32)
    get_observation = np.zeros(nj * 3, dtype=np.float32)

    def get_motor_position(motor_id):
        return get_observation[motor_id]

    def get_motor_velocity(motor_id):
        return get_observation[motor_id + nj]

    def get_motor_torque(motor_id):
        return get_observation[motor_id + nj * 2]

    def set_motor_position(motor_id, posi):
        action_array[motor_id] = posi

    stt = time.time()
    sec = 1.0
    fqt = 1000
    nct = int(fqt * sec)
    for i in range(nct):
        get_observation = np.array(
            stub.GetArray(array_pb2.GetArrayResponse()).data, dtype=np.float32
        )
        sim_time = get_observation[0]
        jq = np.array(np.array2string(get_observation[1 : 1 + nj], separator=", "))
        jv = np.array(
            np.array2string(get_observation[1 + nj : 1 + 2 * nj], separator=", ")
        )
        jf = np.array(
            np.array2string(get_observation[1 + 2 * nj : 1 + 3 * nj], separator=", ")
        )
        print("time:", sim_time)
        print("jq:", jq)
        print("jv:", jv)
        print("jf:", jf)

        action_array[:] = 2.0 * (np.random.random(nj) - 0.5)
        set_request = array_pb2.SetArrayRequest(data=action_array.tolist())

        set_response = stub.SetArray(set_request)
        print("Set method success:", set_response.success)

        ct = time.time()
        time.sleep(max(1.0 / fqt - (ct - stt - i / fqt), 0.0))

    edt = time.time()
    print(
        "Real time: ", np.allclose(edt - stt, sec, atol=2e-3), f" at frequency: {fqt}"
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    nj = 7
    run(nj)
