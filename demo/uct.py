import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pmcts.planners import UCT
from demo.pgm2numpy import read_pgm

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "gray_r"

# Environment extent
extent = [0, 100, 0, 100]
print("Environment extent: ", extent)

# Robot's pose
pose = np.array([90, 20, -np.pi / 2])
print(f"Robot's pose: [x={pose[0]}, y={pose[1]}, yaw={pose[2]}]")

# Read occupancy grid map from pgm file.
occupancy_grid_map = read_pgm("./maps/cave.pgm", byteorder="<")
# Normalize the occupancy grid map.
occupancy_grid_map = (occupancy_grid_map -
                      occupancy_grid_map.min()) / occupancy_grid_map.max()
# Occupancy grid map is now a boolean matrix with 1 indicating occupied.
occupancy_grid_map = occupancy_grid_map < 0.9
assert occupancy_grid_map.dtype == bool

# Create an artificial reward map.
reward_map = np.empty(occupancy_grid_map.shape)
for i in range(reward_map.shape[0]):
    for j in range(reward_map.shape[1]):
        reward_map[i, j] = i * j
# Normalize the reward map.
reward_map = (reward_map - reward_map.min()) / reward_map.max()
reward_map = reward_map.astype(np.float32)

print("Shape of occupancy grid map: ", occupancy_grid_map.shape)
print("Shape of reward map: ", reward_map.shape)
print("Data type of occupancy grid map: ", occupancy_grid_map.dtype)
print("Data type of reward map: ", reward_map.dtype)
print("We used an artificial reward map for a demonstration purpose.")
print("The highest reward is in the upper right corner.")

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
best_action = uct.search(pose, reward_map, occupancy_grid_map)
poses = np.vstack(uct.get_tree())
trajectory = uct.get_trajectory()

print("Blue arrows represent the current best action.")
print("Red arrows show the best trajectory.")
print("Green dots demonstrate the searching tree.")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ogm = axes[0].imshow(occupancy_grid_map, extent=extent)
axes[0].quiver(
    poses[:, 0],
    poses[:, 1],
    np.cos(poses[:, 2]),
    np.sin(poses[:, 2]),
    color="g",
    alpha=0.5,
)
axes[0].quiver(
    best_action[:, 0],
    best_action[:, 1],
    np.cos(best_action[:, 2]),
    np.sin(best_action[:, 2]),
    color="b",
    alpha=0.5,
)
axes[0].quiver(
    trajectory[:, 0],
    trajectory[:, 1],
    np.cos(trajectory[:, 2]),
    np.sin(trajectory[:, 2]),
    color="r",
    alpha=0.1,
)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(ogm, cax=cax)
axes[0].set_title("Occupancy Grid Map")

rm = axes[1].imshow(reward_map, extent=extent)
axes[1].quiver(
    poses[:, 0],
    poses[:, 1],
    np.cos(poses[:, 2]),
    np.sin(poses[:, 2]),
    color="g",
    alpha=0.5,
)
axes[1].quiver(
    best_action[:, 0],
    best_action[:, 1],
    np.cos(best_action[:, 2]),
    np.sin(best_action[:, 2]),
    color="b",
    alpha=0.5,
)
axes[1].quiver(
    trajectory[:, 0],
    trajectory[:, 1],
    np.cos(trajectory[:, 2]),
    np.sin(trajectory[:, 2]),
    color="r",
    alpha=0.1,
)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(rm, cax=cax)
axes[1].set_title("Reward Map")

fig.tight_layout()
plt.show()
