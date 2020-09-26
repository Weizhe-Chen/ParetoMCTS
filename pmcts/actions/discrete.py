"""
Discrete actions sending a specific control command for some period.
"""
import numpy as np
from ..dynamics import Dubins
from ..utilities.matrices import rotation


class DiscreteActions:
    """
    Each action consists of a series of configurations.
    """

    def __init__(self, angle_range, num_actions, duration, velocity=1.0):
        assert len(angle_range) == 2
        self.angle_range = angle_range
        self.num_actions = num_actions
        self.duration = duration
        self.control_inputs = np.linspace(angle_range[0], angle_range[1], num_actions)
        dynamics = Dubins(velocity)
        # Pre-computed primitive actions
        actions = []
        for action_idx in range(num_actions):
            pose = np.zeros(3)
            action = [pose]
            for _ in range(duration):
                pose = dynamics.steering(pose, self.control_inputs[action_idx])
                action.append(pose)
            actions.append(action)
        self.actions = np.asarray(actions)
        assert self.actions.shape[0] == num_actions
        assert self.actions.shape[1] == duration + 1
        assert self.actions.shape[2] == 3

    def get_action(self, pose, action_idx):
        assert pose.ndim == 1
        action = self.actions[action_idx]

        # Rotation
        rotation_mat = rotation(pose[2])
        action = np.matmul(action, rotation_mat)
        action[:, 2] = (action[:, 2] + pose[2]) % (2 * np.pi)

        # Translation
        action[:, 0] += pose[0]
        action[:, 1] += pose[1]
        return action
