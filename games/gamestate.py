from abc import ABC, abstractmethod

win = float("inf")
loss = -float("inf")
draw = 0


class GameState(ABC):
    """
    An abstract base class representing a generic game state.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the game state object.
        """
        pass

    @abstractmethod
    def apply_action(self, action):
        """
        Apply an action to the current game state and return the resulting new state.
        The state of the instance is not altered in this method, i.e. the move is not applied to this gamestate

        :param action: The action to apply.
        :return: The resulting new game state after applying the action.
        """
        pass

    @abstractmethod
    def get_legal_actions(self):
        """
        Get a list of legal actions for the current game state.

        :return: A list of legal actions.
        """
        pass

    @abstractmethod
    def is_terminal(self):
        """
        Check if the current game state is a terminal state, i.e., the game has ended.

        :return: True if the state is terminal, False otherwise.
        """
        pass

    @abstractmethod
    def get_reward(self, player):
        """
        Return the reward of the terminal state in view of player

        :return: The reward value.
        """
        pass

    @abstractmethod
    def is_capture(self, move):
        """
        Check if a move results in a capture of pieces

        Args:
            move (bool): true if the move captures, false otherwise
        """
        pass

    @abstractmethod
    def evaluate_move(self, move):
        """
        Evaluates a move (used for playouts and move-ordering).

        :param move: The move to evaluate.
        :return: a numeric evaluation of the move given the current gamestate
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Generate a string representation of the game

        :returns: String representation of the game
        """
        pass


import numpy as np


def normalize(value, a):
    """
    Normalize value with range [-a,a] to [-1, 1] using tanh

    Args:
        value (float): the value to normalize
        a (float): absolute max
    """
    return np.tanh(value / a)
