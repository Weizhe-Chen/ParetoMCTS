"""
Dubins car dynamics.
"""
import numpy as np


class Dubins:
    """
    Dubins car dynamics with fixed velocity.

    :param velocity: forward velocity, defaults to 1.0
    :type velocity: float, optional
    """

    def __init__(self, velocity=1.0):
        self.velocity = velocity

    @property
    def velocity(self):
        """
        Velocity getter.
        """
        return self.__velocity

    @velocity.setter
    def velocity(self, v):
        """
        Velocity setter.
        """
        assert v > 0.0
        self.__velocity = v

    def steering(self, pose, angle):
        r"""
        Steering the robot given its current pose and the executed steering angle.

        :param pose: position and orientation [$x_1$, $x_2$, $\theta$]
        :type pose: numpy.ndarray
        :param angle: steering angle
        :type angle: float
        """
        assert -np.pi <= angle <= np.pi
        new_pose = pose.copy()
        new_pose[0] = self.velocity * np.cos(pose[2]) + pose[0]
        new_pose[1] = self.velocity * np.sin(pose[2]) + pose[1]
        new_pose[2] = (pose[2] + angle) % (2 * np.pi)
        return new_pose
