import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pmcts.planners import UCT
from demo.pgm2numpy import read_pgm

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "gray_r"


# Environment extent
extent = [0, 100, 0, 100]
print("Environment extent: ", extent)

# Robot's pose
pose = np.array([90, 20, -np.pi / 2])
path = pose
print(f"Robot's pose: [x={pose[0]}, y={pose[1]}, yaw={pose[2]}]")

# Create an artificial reward map.
x, y = np.mgrid[0:100:1, 0:100:1]
pos = np.dstack((x, y))
rv1 = multivariate_normal([80, 50], [[100, 70], [70, 100]])
rv2 = multivariate_normal([50, 50], [[100, -70], [-70, 100]])
rv3 = multivariate_normal([20, 50], [[100, 70], [70, 100]])
reward_map = rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos)

# Normalize the reward map.
reward_map = (reward_map - reward_map.min()) / reward_map.max()
reward_map = reward_map.astype(np.float32)

occupancy_grid_map = np.zeros_like(reward_map).astype(np.bool)

# Planning
angle_range = [-0.1, 0.1]  # Steering angle range
velocity = 1.0
num_actions = 5  # Number of primitive actions. See /tests/imgs/available_actions.png
duration = 10  # Number of contiguration points ([x, y, theta]) per action
weight = 0.3  # Exploration weight. Larger weights will lead to fatter trees.
max_iter = 1000  # Maximum number of tree search iterations
max_rollout = 5  # Maximum number of rollouts or simulations
uct = UCT(
    extent,
    angle_range,
    velocity,
    num_actions,
    duration,
    weight,
    max_iter,
    max_rollout,
)

print("Blue arrows represent the current best action.")
print("Red arrows show the best trajectory.")
print("Green dots demonstrate the searching tree.")

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(200):
    start = time.time()
    best_action = uct.search(pose, reward_map, occupancy_grid_map)
    end = time.time()
    print(f"Planning takes {end-start:.2f} seconds")
    poses = uct.get_tree()
    trajectory = uct.get_trajectory()
    for pose in best_action:
        path = np.vstack((path, pose))
        ax.cla()
        rm = ax.imshow(reward_map, extent=extent)
        ax.plot(path[:, 0], path[:, 1], lw=1, alpha=0.8)
        ax.quiver(
            poses[:, 0],
            poses[:, 1],
            np.cos(poses[:, 2]),
            np.sin(poses[:, 2]),
            color="g",
            alpha=0.3,
            width=0.001,
        )
        ax.quiver(
            trajectory[:, 0],
            trajectory[:, 1],
            np.cos(trajectory[:, 2]),
            np.sin(trajectory[:, 2]),
            color="b",
            alpha=0.3,
        )
        ax.quiver(
            best_action[:, 0],
            best_action[:, 1],
            np.cos(best_action[:, 2]),
            np.sin(best_action[:, 2]),
            color="r",
            alpha=0.3,
        )
        ax.quiver(
            pose[0],
            pose[1],
            np.cos(pose[2]),
            np.sin(pose[2]),
            color="k",
            alpha=0.9,
        )
        ax.set_title("Press ESC to exit")
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.pause(1 / 60)
fig.tight_layout()
plt.show()
