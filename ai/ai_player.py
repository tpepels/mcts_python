import abc

from games.gamestate import GameState


class AIPlayer(abc.ABC):
    @abc.abstractmethod
    def best_action(self, state: GameState) -> tuple[tuple, float]:
        pass

    @abc.abstractmethod
    def print_cumulative_statistics(self) -> str:
        pass
