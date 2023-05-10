import math
import random

from ai.ai_player import AIPlayer


class UCTNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0

    def add_child(self, child):
        self.children.append(child)

    def value(self):
        return self.total_value / self.visits if self.visits != 0 else 0

    def uct(self, exploration_param):
        if self.visits == 0:
            return float("inf")
        exploitation = self.value()
        exploration = exploration_param * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_param):
        return max(self.children, key=lambda child: child.uct(exploration_param))

    def expand(self):
        for action in self.state.get_legal_actions():
            child_state = self.state.apply_action(action)
            child = UCTNode(child_state, parent=self, action=action)
            self.add_child(child)
        return self.children[random.randrange(len(self.children))]

    def rollout(self, evaluation_function):
        current_state = self.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)
        return evaluation_function(current_state, current_state.player)

    def backpropagate(self, value):
        self.visits += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS(AIPlayer):
    def __init__(self, node_class, state, num_simulations, evaluation_function, exploration_param=1.0):
        self.node_class = node_class
        self.state = state
        self.num_simulations = num_simulations
        self.evaluation_function = evaluation_function
        self.exploration_param = exploration_param
        self.root = self.node_class(self.state)

    def run(self):
        for _ in range(self.num_simulations):
            node = self._select_node()
            value = node.rollout(self.evaluation_function)
            node.backpropagate(value)

    def _select_node(self):
        node = self.root
        while not node.state.is_terminal() and len(node.children) == len(node.state.get_legal_actions()):
            node = node.best_child(self.exploration_param)
        if not node.state.is_terminal() and node.children == []:
            return node.expand()
        return node

    def best_action(self):
        self.run()
        return max(self.root.children, key=lambda child: child.visits).action
