import abc

from games.gamestate import GameState


class AIPlayer(abc.ABC):
    @abc.abstractmethod
    def best_action(self, state: GameState):
        pass
