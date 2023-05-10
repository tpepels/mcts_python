import math
import random

from ai.ai_player import AIPlayer

# I've made changes to the uct method and added the select, update, simulate, expand,
# and playout methods in the UCTNode class, as well as updated the run and best_action methods in the MCTS class.

# the evaluate function has been removed from the MCTS class, and the playout method in the UCTNode class now takes an evaluation_function parameter.
# You should pass the evaluation function when calling the playout method in your game loop or experimentation code.

# To implement MCTS-Solver, I've made changes to the UCTNode class by adding the solved and terminal_value attributes, and modifying the update, simulate, and best_action methods.


class UCTNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0
        self.solved = False
        self.terminal_value = None

    def add_child(self, child):
        self.children.append(child)

    def value(self):
        return self.total_value / self.visits if self.visits != 0 else 0

    def uct(self, exploration_param):
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_value / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select(self, exploration_param):
        best_children = []
        best_uct = float("-inf")

        for child in self.children:
            if child.solved:
                return child
            child_uct = child.uct(exploration_param)
            if child_uct > best_uct:
                best_children = [child]
                best_uct = child_uct
            elif child_uct == best_uct:
                best_children.append(child)

        return random.choice(best_children)

    def update(self, reward):
        self.visits += 1
        self.total_value += reward
        if self.solved:
            return
        if self.parent and self.parent.solved:
            self.solved = True
            self.terminal_value = -self.parent.terminal_value

    def simulate(self):
        if self.solved:
            return self.terminal_value

        if self.state.is_terminal():
            self.solved = True
            self.terminal_value = self.state.get_reward()
            return -self.terminal_value

        if not self.children:
            self.expand()

        selected_child = self.select(1.0)
        reward = -selected_child.simulate()
        self.update(reward)
        return reward

    def expand(self):
        for action in self.state.get_legal_actions():
            child_state = self.state.apply_action(action)
            child = UCTNode(child_state, parent=self, action=action)
            self.add_child(child)
        return self.children[random.randrange(len(self.children))]

    def best_action(self):
        return max(self.children, key=lambda child: child.visits).action

    def playout(self, evaluation_function):
        current_state = self.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)
        return evaluation_function(current_state, current_state.player)


class MCTS(AIPlayer):
    def __init__(self, node_class, state, num_simulations, exploration_param=1.0):
        self.node_class = node_class
        self.state = state
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param
        self.root = self.node_class(self.state)

    def run(self):
        for _ in range(self.num_simulations):
            self.root.simulate()

    def best_action(self):
        self.run()
        return max(self.root.children, key=lambda child: child.visits).action
