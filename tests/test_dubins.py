"""
Test Dubins class
"""
import os
import numpy as np
from pmcts.dynamics import Dubins
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


def test_init():
    """
    Test initialization.
    """
    dubins = Dubins()
    assert dubins.velocity == 1.0


def test_setters():
    """
    Test velocity and noise scale setters.
    """
    dubins = Dubins()
    dubins.velocity = 2.0
    assert dubins.velocity == 2.0


def test_steering():
    """
    Test the steering function in some simple scenarios.
    """
    print('Test steering...\n')
    
    if not os.path.exists('./tests/imgs/'):
        os.makedirs('./tests/imgs/')

    dubins = Dubins(velocity=1.0)

    # Forward
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_title('Move Forward')
    pose = np.zeros(3)
    poses = [pose]
    for _ in range(20):
        pose = dubins.steering(pose, 0.0)
        poses.append(pose)
    poses = np.asarray(poses)
    ax.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]))
    fig.tight_layout()
    plt.savefig('./tests/imgs/forward.png', bbox_inches='tight')

    # Turn left
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_title('Turn Left')
    pose = np.zeros(3)
    poses = [pose]
    for _ in range(20):
        pose = dubins.steering(pose, 0.1)
        poses.append(pose)
    poses = np.asarray(poses)
    ax.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]))
    fig.tight_layout()
    plt.savefig('./tests/imgs/left.png', bbox_inches='tight')

    # Turn right
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_title('Turn Right')
    pose = np.zeros(3)
    poses = [pose]
    for _ in range(20):
        pose = dubins.steering(pose, -0.1)
        poses.append(pose)
    poses = np.asarray(poses)
    ax.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]))
    fig.tight_layout()
    plt.savefig('./tests/imgs/right.png', bbox_inches='tight')
    
    print('Results are saved in /tests/imgs/')
    