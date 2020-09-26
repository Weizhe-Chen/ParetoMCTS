"""
Test DiscreteActions class.
"""
import os
import numpy as np
from pmcts.actions import DiscreteActions
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt


def test_init():
    """
    DiscreteActions should create the default actions after initialization.
    """

    print("Test DiscreteActions.__init__...\n")

    if not os.path.exists('./tests/imgs/'):
        os.makedirs('./tests/imgs/')

    actor = DiscreteActions(angle_range=[-0.1, 0.1], num_actions=5, duration=10, velocity=1.0)
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_title("Default Actions")
    for action in actor.actions:
        ax.quiver(action[:, 0], action[:, 1], np.cos(action[:, 2]), np.sin(action[:, 2]))
    fig.tight_layout()
    plt.savefig('./tests/imgs/default_actions.png', bbox_inches='tight')


def test_get_action():
    print("Test DiscreteActions.get_action()...\n")

    if not os.path.exists('./tests/imgs/'):
        os.makedirs('./tests/imgs/')

    pose = 5 * np.ones(3)
    pose[2] = np.pi / 2

    actor = DiscreteActions(angle_range=[-0.1, 0.1], num_actions=5, duration=10, velocity=1.0)
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_title("Available Actions at [5, 5, pi/2]")
    for action_idx in range(5):
        action = actor.get_action(pose, action_idx)
        ax.quiver(action[:, 0], action[:, 1], np.cos(action[:, 2]), np.sin(action[:, 2]))
    fig.tight_layout()
    plt.savefig('./tests/imgs/available_actions.png', bbox_inches='tight')