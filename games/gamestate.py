# cython: language_level=3

from operator import itemgetter
from typing import Generator
import cython
from abc import ABC, abstractmethod

win: cython.int = 9999999
loss: cython.int = -9999999
draw: cython.int = 0


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
    def apply_action(self, action) -> GameState:
        """
        Apply an action to the current game state and return the resulting new state.
        The state of the instance is not altered in this method, i.e. the move is not applied to this gamestate

        :param action: The action to apply.
        :return: The resulting new game state after applying the action.
        """
        pass

    @abstractmethod
    def get_random_action(self) -> tuple:
        """
        Return a single legal action, uniformly chosen from all legal actions
        """
        pass

    @abstractmethod
    def yield_legal_actions(self) -> Generator[tuple, None, None]:
        """
        Returns legal actions one by one
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> list:
        """
        Get a list of legal actions for the current game state.

        :return: A list of legal actions.
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the current game state is a terminal state, i.e., the game has ended.

        :return: True if the state is terminal, False otherwise.
        """
        pass

    @abstractmethod
    def get_reward(self, player) -> float:
        """
        Return the reward of the terminal state in view of player

        :return: The reward value. win, loss or draw
        """
        pass

    @abstractmethod
    def is_capture(self, move) -> bool:
        """
        Check if a move results in a capture of pieces

        Args:
            move (bool): true if the move captures, false otherwise
        """
        pass

    @abstractmethod
    def evaluate_moves(self, moves) -> list[tuple[tuple, int]]:
        """Evaluate a list of moves to use in move ordering

        Args:
            moves list: A list of moves to evaluate
        """
        pass

    @abstractmethod
    def move_weights(self, moves) -> list[int]:
        """Evaluate a list of moves to use in move ordering

        Args:
            moves list: A list of moves to evaluate
        """
        pass

    @abstractmethod
    def evaluate_move(self, move) -> int:
        """
        Evaluates a move use for playouts. (This is a simple evaluation)

        :param move: The move to evaluate.
        :return: a numeric evaluation of the move given the current gamestate
        """
        pass

    @abstractmethod
    def visualize(self, full_debug=False) -> str:
        """
        Generate a string representation of the game

        :returns: String representation of the game
        """
        pass

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        pass


import numpy as np
import random


@cython.ccall
@cython.locals(value=cython.double, a=cython.double)
def normalize(value, a):
    """
    Normalize value with range [-a,a] to [-1, 1] using tanh

    Args:
        value (float): the value to normalize
        a (float): absolute max
    """
    return np.tanh(value / a)
