import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, output_max=1e6, integrator_max=1e6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_max = output_max
        self.error_sum = 0
        self.last_error = 0
        self.integrator_max = integrator_max

    def output(self, error, dt=1.0):
        self.error_sum += error * dt
        self.error_sum = np.clip(
            self.error_sum, -self.integrator_max, self.integrator_max
        )
        error_diff = (error - self.last_error) / dt
        out = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        self.last_error = error
        return np.clip(out, -self.output_max, self.output_max)

    def output_d(self, error, d_error, dt=1.0):
        self.error_sum += error * dt
        self.error_sum = np.clip(
            self.error_sum, -self.integrator_max, self.integrator_max
        )
        out = self.kp * error + self.ki * self.error_sum + self.kd * d_error
        return np.clip(out, -self.output_max, self.output_max)

    def reset(self):
        self.error_sum = 0
        self.last_error = 0


class PIDarray:
    def __init__(self, kps, kis, kds, integrator_maxs=None):
        assert (
            len(kps) == len(kis) == len(kds)
        ), "kps, kis, kds must have the same length"
        self.kps = kps
        self.kis = kis
        self.kds = kds
        self.integrator_maxs = integrator_maxs
        self.error_sums = np.zeros_like(kps)
        self.last_errors = np.zeros_like(kps)

    def output(self, errors, dt=1.0):
        self.error_sums += errors * dt
        if not self.integrator_maxs is None:
            self.error_sums = np.clip(
                self.error_sums, -self.integrator_maxs, self.integrator_maxs
            )
        error_diffs = (errors - self.last_errors) / dt
        outs = self.kps * errors + self.kis * self.error_sums + self.kds * error_diffs
        self.last_errors = errors
        return outs

    def reset(self):
        self.error_sums = np.zeros_like(self.kps)
        self.last_errors = np.zeros_like(self.kps)
