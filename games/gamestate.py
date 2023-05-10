from abc import ABC, abstractmethod


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
    def get_reward(self):
        """
        The reward is 1 for player 1 if they have won, -1 for player 2 if they have won, and 0 otherwise.

        :return: The reward value.
        """
        pass
