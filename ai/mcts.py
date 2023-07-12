import random
import time
from typing import List, Tuple

import numpy as np
from ai.ai_player import AIPlayer
from ai.transpos_table import TranspositionTableMCTS
from games.gamestate import GameState, win, loss, draw


class Node:
    def __init__(self, state: GameState, action: Tuple, tt: TranspositionTableMCTS, c: float = 1.0):
        self.state = state
        self.children: List[Node] = []
        # The action that led to this state, needed for root selection
        self.action = action
        self.tt = tt
        self.c = c
        self.player = state.player
        # This checks if in another part of the tree, this state was already expanded.
        # In that case, we can immediately add all children and mark the node as expanded
        stats = self.stats()
        self.expanded = stats[5]
        if self.expanded:
            self.add_all_children()
        else:
            self.expanded = False

    def select(self):
        if not self.expanded:
            return self.expand()
        else:
            return self.utc()

    def utc(self):
        _, _, _, my_visits, _, _ = self.stats()

        max_val = -float("inf")
        max_child = None

        for child in self.children:
            # 0: v1, 1: v2, 2: im_value, 3: visits, 4: solved_player, 5: is_expanded
            stats = child.stats()
            i = self.player - 1  # This makes is easier to index the player in the stats

            uct_val = (
                stats[i] / stats[3]
                + np.sqrt(self.c * np.log(my_visits) / stats[3])
                + np.random.uniform(0.0001, 0.001)
            )
            if uct_val >= max_val:
                max_child = child
                max_val = uct_val

        return max_child

    def expand(self):
        for action in self.state.yield_legal_actions():
            if action not in [child.action for child in self.children]:
                child = Node(self.state.apply_action(action), action=action, tt=self.tt, c=self.c)
                self.children.append(child)
                return child

        # The node is fully expanded so we can switch to UTC selection
        self.expanded = True
        self.tt.put(self.state.board_hash, is_expanded=True, board=self.state.board)
        return self.utc()

    def add_all_children(self):
        actions = self.state.get_legal_actions()
        for action in actions:
            child = Node(self.state.apply_action(action), action=action, tt=self.tt, c=self.c)
            self.children.append(child)

    def stats(self):
        # 0: v1, 1: v2, 3: im_value, 4: visits, 5: solved_player, 6: is_expanded
        return self.tt.get(self.state.board_hash, board=self.state.board)

    def __str__(self):
        return f"Node(Action: {self.action}, C: {self.c}, Player: {self.player}, Expanded: {self.expanded})"


class MCTSPlayer(AIPlayer):
    def __init__(
        self,
        player: int,
        evaluate,
        transposition_table_size=2**16,
        num_simulations=None,
        max_time=None,
        c=1.0,
        debug=False,
    ):
        self.player = player
        self.evaluate = evaluate
        if num_simulations:
            self.num_simulations = num_simulations
            self.time = None
        elif max_time:
            self.time = max_time
            self.num_simulations = None
        else:
            assert (
                time is not None or num_simulations is not None
            ), "Either provide num_simulations or search time"

        self.debug = debug
        self.c = c
        self.tt = TranspositionTableMCTS(transposition_table_size)

    def best_action(self, state: GameState):
        self.root = Node(state, None, self.tt, c=self.c)  # reset root for new round of MCTS

        if self.num_simulations:
            for _ in range(self.num_simulations):
                self.simulate()
        else:
            start_time = time.time()
            while time.time() - start_time < self.time:
                self.simulate()

        # Clean the transposition table
        self.tt.evict()
        max_node = max(self.root.children, key=lambda node: node.stats()[3])
        return max_node.action, 0  # return the most visited state

    def simulate(self):
        node = self.root
        selected = [node]

        while not node.state.is_terminal():
            _node = node.select()
            selected.append(_node)
            node = _node

        # Do a random playout and collect the result
        result = self.play_out(node.state)

        # Backpropagate the result along the chosen nodes
        for node in selected:
            self.tt.put(
                key=node.state.board_hash, v1=result[0], v2=result[1], visits=1, board=node.state.board
            )

    def play_out(self, state: GameState):
        while not state.is_terminal():
            action = state.get_random_action()
            state = state.apply_action(action)

        # Map the result to the correct player
        result = state.get_reward(1)
        if result == win:
            result = (1, 0)
        elif result == loss:
            result = (0, 1)
        else:
            result = (0, 0)
        return result

    def print_cumulative_statistics(self) -> str:
        pass
