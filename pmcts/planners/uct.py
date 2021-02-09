"""
Upper confidence bound applied to Monte-Carlo tree search (UCT).
"""
from copy import copy
import numpy as np

from ..actions import DiscreteActions
from ..utilities.indexing import xy_to_ij


class Node:
    def __init__(self, pose, num_actions, reward=0.0, action=None, parent=None):
        self.pose = pose
        self.unvisited_actions = list(range(num_actions))
        self.reward = reward
        self.action = action
        self.parent = parent
        self.children = list()
        self.num_visits = 0

    def add_child(self, node):
        self.children.append(node)


class UCT:
    def __init__(
        self,
        extent,
        angle_range,
        velocity,
        num_actions,
        duration,
        weight,
        max_iter,
        max_rollout,
    ):
        self.extent = extent
        self.num_actions = num_actions
        # Number of actions must be an odd number
        assert self.num_actions % 2 != 0
        self.weight = weight
        self.max_iter = max_iter
        self.max_rollout = max_rollout
        self.actor = DiscreteActions(angle_range, num_actions, duration, velocity)
        self.reward = None
        self.occupancy = None
        self.max_row = None
        self.max_col = None
        self.root = None

    def search(self, pose, reward_map, occupancy_map):
        # Save reward map and occupancy map
        self.reward = reward_map
        self.occupancy = occupancy_map
        assert self.occupancy.dtype == bool
        self.max_row = self.reward.shape[0] - 1
        self.max_col = self.reward.shape[1] - 1

        # Initialize root node
        self.root = Node(pose, self.num_actions)
        # MCTS main loop
        for _ in range(self.max_iter):
            # Selection
            expandable_node, has_valid_child = self.select(self.root)
            # This branch is blocked by obstacles.
            if not has_valid_child:
                self.backpropagation(expandable_node, reward=-1.0)
                break

            # Expansion
            new_node = self.expand(expandable_node)

            # Simulation / rollout
            if new_node is None:
                # No valid action available.
                reward = -1.0
            else:
                reward = self.rollout(new_node)

            # Backpropagation
            if new_node is None:
                self.backpropagation(expandable_node, reward)
            else:
                self.backpropagation(new_node, reward)

        return self.best_action(self.root)

    def select(self, node):
        # Select this node if it has some unvisited actions
        if len(node.unvisited_actions) != 0:
            return node, True
        # If all actions have been visited but the children dict is empty
        elif not node.children:
            return node, False
        else:  # All actions have been visited and the children dict is not empty
            exploitation_scores = [
                child.reward / child.num_visits for child in node.children
            ]
            exploration_scores = [
                np.sqrt(2.0 * np.log(node.num_visits) / child.num_visits)
                for child in node.children
            ]
            # Note that we rescaled the exploration weight
            # according to the maximum exploitation score.
            explore_weight = np.max(exploitation_scores) * self.weight
            upper_confidence_bounds = [
                exploit + explore_weight * explore
                for exploit, explore in zip(exploitation_scores, exploration_scores)
            ]
            index = np.argmax(upper_confidence_bounds)
            selected_node, has_valid_child = self.select(node.children[index])
            return selected_node, has_valid_child

    def boundary_check(self, action):
        if np.any(action[:, 0] <= self.extent[0]):
            return False
        if np.any(action[:, 0] >= self.extent[1]):
            return False
        if np.any(action[:, 1] <= self.extent[2]):
            return False
        if np.any(action[:, 1] >= self.extent[3]):
            return False
        return True

    def collision_check(self, ij):
        occupied = self.occupancy[ij[:, 0], ij[:, 1]]
        if np.any(occupied):
            return False
        return True

    def expand(self, parent):
        # Randomly select an action from the available actions
        action_idx = parent.unvisited_actions[0]
        del parent.unvisited_actions[0]
        action = self.actor.get_action(parent.pose, action_idx)

        # Check whether this action is valid
        ij = xy_to_ij(action[:, :2], self.extent, self.max_row, self.max_col)
        is_inside_boundary = self.boundary_check(action)
        is_outside_obstacle = self.collision_check(ij)
        is_valid_action = is_inside_boundary and is_outside_obstacle

        # Create the child node and attach it to its parent
        child = None
        if is_valid_action:
            pose = action[-1]
            rewards = self.reward[ij[:, 0], ij[:, 1]]
            reward = np.sum(rewards, axis=0)
            child = Node(pose, self.num_actions, reward, action, parent)
            parent.add_child(child)
        return child

    def rollout(self, node):
        # Default policy is moving forward
        action_idx = self.num_actions // 2
        pose = copy(node.pose)
        poses = []
        for _ in range(self.max_rollout):
            action = self.actor.get_action(pose, action_idx)
            pose = action[-1]
            poses.append(action)
        poses = np.vstack(poses)
        ij = xy_to_ij(poses[:, :2], self.extent, self.max_row, self.max_col)
        rewards = self.reward[ij[:, 0], ij[:, 1]]
        reward = np.sum(rewards, axis=0)
        average_reward = reward / self.max_rollout
        return average_reward

    def backpropagation(self, node, reward):
        node.reward += reward
        node.num_visits += 1
        if node.parent is not None:
            self.backpropagation(node.parent, reward)

    def best_action(self, node):
        assert len(node.unvisited_actions) == 0
        if not node.children:
            raise ValueError(
                "No valid action in current pose!\n"
                "You might need to implement some 'turn around' engineering "
                "tricks to solve this problem."
            )
        num_visits = [child.num_visits for child in node.children]
        # print(sorted(num_visits))
        idx = np.argmax(num_visits)
        best_child = node.children[idx]
        return best_child.action

    def get_trajectory(self):
        assert len(self.root.unvisited_actions) == 0
        if not self.root.children:
            raise ValueError(
                "No valid action in current pose!\n"
                "You might need to implement some 'turn around' engineering "
                "tricks to solve this problem."
            )
        poses = []
        node = copy(self.root)
        while node.children:
            num_visits = [child.num_visits for child in node.children]
            idx = np.argmax(num_visits)
            best_child = node.children[idx]
            poses.append(best_child.action)
            node = best_child
        return np.vstack(poses)

    def get_tree_resursive(self, node, poses):
        poses.append(node.action)
        if node.children:
            for child in node.children:
                poses = self.get_tree_resursive(child, poses)
        return poses

    def get_tree(self):
        poses = []
        for child in self.root.children:
            poses = self.get_tree_resursive(child, poses)
        return np.vstack(poses)
