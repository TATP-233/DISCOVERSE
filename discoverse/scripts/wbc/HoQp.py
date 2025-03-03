from Task import Task

import numpy as np
from scipy.linalg import null_space

try:
    from qpsolvers import solve_qp
except ImportError:
    error_msg = "\n------------------------------------------------------------------------------\n"
    error_msg += "qpsolvers not found, try:\n"
    error_msg += (
        "pip install qpsolvers[quadprog] -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
    )
    error_msg += "------------------------------------------------------------------------------\n"
    raise ImportError(error_msg)


class HoQp:
    def __init__(self, task: Task, higherProblem=None) -> None:
        self.task_ = task
        self.higherProblem_ = higherProblem
        self.initVars()
        self.formulateProblem()
        self.solveProblem()
        # For next problem
        self.buildZMatrix()
        self.stackSlackSolutions()

    @property
    def getStackedZMatrix(self):
        return self.stackedZ_

    @property
    def getStackedTasks(self):
        return self.stackedTasks_

    @property
    def getStackedSlackSolutions(self):
        return self.stackedSlackVars_

    @property
    def getSlackedNumVars(self):
        return self.stackedTasks_.d_.shape[0]

    def getSolutions(self):
        return self.xPrev_ + np.dot(self.stackedZPrev_, self.decisionVarsSolutions_)

    def initVars(self):
        # Task variables
        self.numSlackVars_ = self.task_.d_.shape[0]
        self.hasEqConstraints_ = self.task_.a_.shape[0] > 0
        self.hasIneqConstraints_ = self.numSlackVars_ > 0

        # Pre-Task variables
        if self.higherProblem_ is not None:
            self.stackedZPrev_ = self.higherProblem_.getStackedZMatrix
            self.stackedTasksPrev_ = self.higherProblem_.getStackedTasks
            self.stackedSlackSolutionsPrev_ = (
                self.higherProblem_.getStackedSlackSolutions
            )
            self.xPrev_ = self.higherProblem_.getSolutions()
            self.numPrevSlackVars_ = self.higherProblem_.getSlackedNumVars
            self.numDecisionVars_ = self.stackedZPrev_.shape[1]

        else:
            self.numDecisionVars_ = max(self.task_.a_.shape[1], self.task_.d_.shape[1])
            self.stackedTasksPrev_ = Task(
                None, None, None, None, numDecisionVars=self.numDecisionVars_
            )
            self.stackedZPrev_ = np.eye(self.numDecisionVars_)
            self.stackedSlackSolutionsPrev_ = np.zeros((0, 1))
            self.xPrev_ = np.zeros((self.numDecisionVars_, 1))
            self.numPrevSlackVars_ = 0

        self.stackedTasks_ = self.task_ + self.stackedTasksPrev_

        # Init convenience matrices
        self.eyeNvNv_ = np.eye(self.numSlackVars_)
        self.zeroNvNx_ = np.zeros((self.numSlackVars_, self.numDecisionVars_))

    def formulateProblem(self):
        self.buildHMatrix()
        self.buildCVector()
        self.buildDMatrix()
        self.buildFVector()

    def buildHMatrix(self):
        zTaTaz = np.zeros((self.numDecisionVars_, self.numDecisionVars_))
        if self.hasEqConstraints_:
            # Make sure that all eigenvalues of A_t_A are non-negative, which could arise due to numerical issues
            aCurrZPrev = np.dot(self.task_.a_, self.stackedZPrev_)
            zTaTaz = np.dot(aCurrZPrev.T, aCurrZPrev) + 1e-12 * np.eye(
                self.numDecisionVars_
            )
            # This way of splitting up the multiplication is about twice as fast as multiplying 4 matrices
        self.h_ = np.vstack(
            (
                np.hstack((zTaTaz, self.zeroNvNx_.T)),
                np.hstack((self.zeroNvNx_, self.eyeNvNv_)),
            )
        )

    def buildCVector(self):
        zeroVec = np.zeros((self.numSlackVars_, 1))
        temp = np.zeros((self.numDecisionVars_, 1))
        if self.hasEqConstraints_:
            temp = np.dot(
                np.dot(self.task_.a_, self.stackedZPrev_).T,
                (np.dot(self.task_.a_, self.xPrev_) - self.task_.b_),
            )
        self.c_ = np.vstack((temp, zeroVec))

    def buildDMatrix(self):
        stackedZero = np.zeros((self.numPrevSlackVars_, self.numSlackVars_))

        dCurrZ = np.zeros((0, self.numDecisionVars_))
        if self.hasIneqConstraints_:
            dCurrZ = np.dot(self.task_.d_, self.stackedZPrev_)

        # NOTE: This is upside down compared to the paper,
        # but more consistent with the rest of the algorithm
        self.d_ = np.vstack(
            (
                np.hstack((self.zeroNvNx_, -self.eyeNvNv_)),
                np.hstack(
                    (np.dot(self.stackedTasksPrev_.d_, self.stackedZPrev_), stackedZero)
                ),
                np.hstack((dCurrZ, -self.eyeNvNv_)),
            )
        )

    def buildFVector(self):
        zeroVec = np.zeros((self.numSlackVars_, 1))

        fMinusDXPrev = np.zeros((0, 1))
        if self.hasIneqConstraints_:
            fMinusDXPrev = self.task_.f_ - np.dot(self.task_.d_, self.xPrev_)

        self.f_ = np.vstack(
            (
                zeroVec,
                self.stackedTasksPrev_.f_
                - np.dot(self.stackedTasksPrev_.d_, self.xPrev_)
                + self.stackedSlackSolutionsPrev_,
                fMinusDXPrev,
            )
        )

    def buildZMatrix(self):
        if self.hasEqConstraints_:
            assert self.task_.a_.shape[1] > 0
            self.stackedZ_ = np.dot(
                self.stackedZPrev_,
                null_space(np.dot(self.task_.a_, self.stackedZPrev_)),
            )
        else:
            self.stackedZ_ = self.stackedZPrev_

    def solveProblem(self):
        # ['cvxopt', 'ecos', 'osqp', 'quadprog', 'scs']
        sol = solve_qp(self.h_, self.c_.T[0], self.d_, self.f_.T[0], solver="quadprog")
        self.decisionVarsSolutions_ = sol[: self.numDecisionVars_].reshape(
            (self.numDecisionVars_, 1)
        )
        if self.numSlackVars_:
            self.slackVarsSolutions_ = sol[-self.numSlackVars_ :].reshape(
                (self.numSlackVars_, 1)
            )
        else:
            self.slackVarsSolutions_ = np.zeros((0, 1))

    def stackSlackSolutions(self):
        if self.higherProblem_ is not None:
            self.stackedSlackVars_ = np.vstack(
                (self.higherProblem_.getStackedSlackSolutions, self.slackVarsSolutions_)
            )
        else:
            self.stackedSlackVars_ = self.slackVarsSolutions_


if __name__ == "__main__":
    a = np.array(
        [
            [0.680375, 0.566198, 0.823295, -0.329554],
            [-0.211234, 0.59688, -0.604897, 0.536459],
        ]
    )
    b = np.array([[1], [1]])
    d = np.array(
        [
            [-0.444451, -0.0452059, -0.270431, 0.904459],
            [0.10794, 0.257742, 0.0268018, 0.83239],
        ]
    )
    f = np.array([[1], [1]])
    # task0 = Task(a, b, d, f)
    task0 = Task(a, b, d, f)

    a1 = np.ones((2, 4))
    task1 = Task(a1, b, d, f)

    hoQp0 = HoQp(task0)
    hoQp1 = HoQp(task1, hoQp0)

    x0 = hoQp0.getSolutions()
    print(a @ x0 - b)
    print(d @ x0 - f)

    x1 = hoQp1.getSolutions()
    print(a @ x1 - b)
    print(d @ x1 - f)

    print(a1 @ x1 - b)

    print(x0.T[0])
    print(x1.T[0])
