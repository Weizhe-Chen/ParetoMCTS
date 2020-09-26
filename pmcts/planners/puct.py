import numpy as np
from numpy.random import choice
from .uct import UCT


def build_pareto_front(dic):
    """
    Efficient Pareto front computation for bi-objective case.
    """
    id_vector = list(dic.items())
    for i in range(len(id_vector)):
        for j in range(len(id_vector)):
            if i == j:
                continue
            arr_i = np.array(id_vector[i][1])
            arr_j = np.array(id_vector[j][1])
            if np.all(arr_i <= arr_j) and np.any(arr_i < arr_j):
                del dic[id_vector[i][0]]
                break
    return dic


class ParetoUCT(UCT):
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
        super(ParetoUCT, self).__init__(
            extent,
            angle_range,
            velocity,
            num_actions,
            duration,
            weight,
            max_iter,
            max_rollout,
        )

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
            # Equation (3) in the paper
            exploration_scores = [
                np.sqrt(
                    (4.0 * np.log(node.num_visits) + np.log(len(child.reward)))
                    / (2.0 * child.num_visits)
                )
                for child in node.children
            ]
            explore_weight = np.max(exploitation_scores, axis=0) * self.weight
            upper_confidence_bounds = {
                idx: exploit + explore_weight * explore
                for idx, (exploit, explore) in enumerate(
                    zip(exploitation_scores, exploration_scores)
                )
            }
            pareto_front = build_pareto_front(upper_confidence_bounds)
            index = choice(list(pareto_front.keys()))
            selected_node, has_valid_child = self.select(node.children[index])
            return selected_node, has_valid_child
