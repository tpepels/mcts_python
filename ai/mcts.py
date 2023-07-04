import math
from random import random

from ai.ai_player import AIPlayer
from games.gamestate import GameState

# I've made changes to the uct method and added the select, update, simulate, expand,
# and playout methods in the UCTNode class, as well as updated the run and best_action methods in the MCTS class.

# To implement MCTS-Solver, I've made changes to the UCTNode class by adding the solved and terminal_value attributes,
# and modifying the update, simulate, and best_action methods.


class UCTNode:
    def __init__(self, state: GameState, parent=None, action=None, c=1.0):
        """
        Initialize a UCT (Upper Confidence Bound for Trees) node.

        :param state: The game state at this node.
        :param parent: The parent node.
        :param action: The action taken to reach this node.
        """
        self.state: GameState = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0
        self.solved = False
        self.terminal_value = None
        self.c: float = c

    def simulate(self):
        """
        Perform a simulation from this node ending in a playout from this node.

        :return: The reward from the simulation.
        """
        if self.solved:
            return self.terminal_value

        if self.state.is_terminal():
            self.solved = True
            self.terminal_value = self.state.get_reward()  # TODO Rewards are in absolute perspective of p1
            return -self.terminal_value

        if not self.children:
            self.expand()

        selected_child = self.select()
        reward = -selected_child.playout()
        self.update(reward)
        return reward

    def uct(self) -> float:
        """
        Calculate the UCT value of this node.

        :return: The UCT value.
        """
        if self.visits == 0:
            return math.inf

        exploitation = self.total_value / self.visits
        exploration = self.c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select(self):
        """
        Select the best child based on UCT value.

        :return: The selected child node.
        """
        best_children = []
        best_uct = float("-inf")

        for child in self.children:
            # TODO Dit klopt niet, het ligt eraan vanuit welk perspectief je kijkt
            if child.solved:
                return child

            child_uct = child.uct()
            if child_uct > best_uct:
                best_children = [child]
                best_uct = child_uct
            elif child_uct == best_uct:
                best_children.append(child)

        return random.choice(best_children)

    def update(self, reward):
        """
        Update the visits and total value of this node.

        :param reward: The reward obtained from the simulation.
        """
        self.visits += 1
        self.total_value += reward

        if self.solved:
            return

        if self.parent and self.parent.solved:
            self.solved = True
            self.terminal_value = -self.parent.terminal_value

    def expand(self):
        """
        Expand this node by generating all possible child nodes.

        :return: A randomly selected child node for playout.
        """
        for action in self.state.get_legal_actions():
            child_state = self.state.apply_action(action)
            child = UCTNode(child_state, parent=self, action=action)
            self.add_child(child)

        # Return a random node for playout
        return self.children[random.randrange(len(self.children))]

    def best_action(self):
        """
        Determine the best action based on the most visited child node.

        :return: The action corresponding to the most visited child node.
        """
        return max(self.children, key=lambda child: child.visits).action

    def playout(self):
        """
        Perform a - random - playout from this node.

        :param evaluation_function: The evaluation function to score the resulting state.
        :return: The score of the final state.
        """
        current_state = self.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)

        return self.state.get_reward()

    def add_child(self, child):
        """
        Add a child node.

        :param child: The child node to add.
        """
        self.children.append(child)

    def value(self):
        """
        Calculate the value of this node.

        :return: The average value of this node.
        """
        return self.total_value / self.visits if self.visits != 0 else 0


class MCTSPlayer(AIPlayer):
    def init(self, node_class, state, num_simulations, exploration_param=1.0):
        """
        Initialize a Monte Carlo Tree Search (MCTS) player.

        :param node_class: The UCTNode class to be used for nodes.
        :param state: The current game state.
        :param num_simulations: The number of simulations to run.
        :param exploration_param: The exploration parameter.
        """
        self.node_class = node_class
        self.state = state
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param
        self.root: UCTNode = self.node_class(self.state)

    def run(self):
        """
        Run the MCTS algorithm for the specified number of simulations.
        """
        for _ in range(self.num_simulations):
            self.root.simulate()

    def best_action(self):
        """
        Determine the best action based on the most visited child node after running MCTS.

        :return: The best action.
        """
        self.run()
        return self.root.best_action()

    def print_cumulative_statistics(self) -> str:
        pass
