import abc
import random
import math


class AIPlayer(abc.ABC):
    def __init__(self, state):
        self.state = state

    @abc.abstractmethod
    def best_action(self):
        pass
