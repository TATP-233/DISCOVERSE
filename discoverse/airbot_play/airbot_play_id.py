import numpy as np
import pinocchio as pin


class AirbotPlayID:
    def __init__(self, urdf) -> None:
        self.pin_model = pin.buildModelFromUrdf(urdf)
        self.pin_data = self.pin_model.createData()

    def inverseDyn(self, q, v):
        pin.forwardKinematics(self.pin_model, self.pin_data, q, v, np.zeros(6))
        pin.nonLinearEffects(self.pin_model, self.pin_data, q, v)
        return self.pin_data.nle


if __name__ == "__main__":
    import os

    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    airbot_play_id = AirbotPlayID(
        os.path.join(
            os.path.dirname(__file__),
            "../../models/urdf/airbot_play_v3_gripper_fixed.urdf",
        )
    )

    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau = airbot_play_id.inverseDyn(q, v)

    print(tau)
