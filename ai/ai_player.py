import abc


class AIPlayer(abc.ABC):
    @abc.abstractmethod
    def best_action(self):
        pass
