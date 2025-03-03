import numpy as np
import pinocchio as pin


class Abt_HOQP:
    info_ = {}
    jointVelLimits_ = 1.0  # rad/s

    def __init__(self, urdf_path) -> None:
        self.pin_model_ = pin.buildModelFromUrdf(urdf_path)
        self.pin_data_ = self.pin_model_.createData()

        self.info_["endEffectorFrameIndex"] = self.pin_model_.getFrameId(
            "joint_custom_end"
        )
        self.info_["actuatedDofNum"] = 6
        self.info_["numDecisionVars"] = 6
        self.tld_ = np.zeros(
            (2 * self.info_["actuatedDofNum"], self.info_["numDecisionVars"])
        )
        self.tld_[
            : self.info_["actuatedDofNum"], -self.info_["actuatedDofNum"] :
        ] = np.eye(self.info_["actuatedDofNum"])
        self.tld_[
            self.info_["actuatedDofNum"] :, -self.info_["actuatedDofNum"] :
        ] = -np.eye(self.info_["actuatedDofNum"])
        self.tlf_ = self.jointVelLimits_ * np.ones(
            (2 * self.info_["actuatedDofNum"], 1)
        )

    def getJac(self, q):
        pin.forwardKinematics(self.pin_model_, self.pin_data_, q)
        pin.computeJointJacobians(self.pin_model_, self.pin_data_)
        jac = pin.getFrameJacobian(
            self.pin_model_,
            self.pin_data_,
            self.info_["endEffectorFrameIndex"],
            pin.LOCAL_WORLD_ALIGNED,
        ).copy()
        return jac
