import numpy as np


class Task:
    def __init__(self, a, b, d, f, numDecisionVars=0):
        if numDecisionVars:
            self.a_ = np.zeros((0, numDecisionVars))
            self.b_ = np.zeros((0, 1))
            self.d_ = np.zeros((0, numDecisionVars))
            self.f_ = np.zeros((0, 1))
        else:
            if a is None:
                self.a_ = np.zeros((0, d.shape[1]))
                self.b_ = np.zeros((0, 1))
            else:
                self.a_ = a
                self.b_ = b

            if d is None:
                self.d_ = np.zeros((0, a.shape[1]))
                self.f_ = np.zeros((0, 1))
            else:
                self.d_ = d
                self.f_ = f

    def __add__(self, other):
        return Task(
            np.vstack((self.a_, other.a_)),
            np.vstack((self.b_, other.b_)),
            np.vstack((self.d_, other.d_)),
            np.vstack((self.f_, other.f_)),
        )
