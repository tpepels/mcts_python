import itertools
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

            # TODO Dit moet je nog checken
            if stats[4] is not None:  # This child node has been solved
                if stats[4] == self.player:  # Winning move
                    self.solved = True
                    self.tt.put(
                        self.state.board_hash, is_expanded=True, solved=self.player, board=self.state.board
                    )
                    return child
                else:  # Losing move
                    continue  # Skip this child

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
        # TODO Dit moet je nog checken
        all_children_loss = True
        for action in self.state.yield_legal_actions():
            if action not in [child.action for child in self.children]:
                child = Node(self.state.apply_action(action), action=action, tt=self.tt, c=self.c)
                self.children.append(child)
                if not child.solved or child.stats()[4] != self.player:  # This child is not a loss
                    all_children_loss = False
                elif child.stats()[4] == self.player:  # This child is a win
                    self.solved = True
                    self.tt.put(
                        self.state.board_hash, is_expanded=True, solved=self.player, board=self.state.board
                    )
                    return child

        if all_children_loss:  # All children are loss
            self.solved = True
            self.tt.put(
                self.state.board_hash, is_expanded=True, solved=3 - self.player, board=self.state.board
            )

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
        transposition_table_size: int = 2**16,
        num_simulations: int = None,
        max_time: int = None,
        c: float = 1.0,
        dyn_early_term: bool = False,
        dyn_early_term_cutoff: float = 0.9,
        early_term: bool = False,
        early_term_turns: int = 10,
        e_greedy: bool = False,
        e_g_epsilon: float = 0.05,
        e_g_subset: int = 20,
        node_priors: bool = False,
        debug: bool = False,
    ):
        self.player = player
        self.evaluate = evaluate

        self.dyn_early_term = dyn_early_term
        self.dyn_early_term_cutoff = dyn_early_term_cutoff
        self.e_greedy = e_greedy
        self.e_g_epsilon = e_g_epsilon
        self.e_g_subset = e_g_subset
        self.early_term = early_term
        self.early_term_turns = early_term_turns

        self.node_priors = node_priors
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
            node = node.select()
            selected.append(node)

        # Do a random playout and collect the result
        result = self.play_out(node.state)

        # Backpropagate the result along the chosen nodes
        for node in selected:
            self.tt.put(
                key=node.state.board_hash, v1=result[0], v2=result[1], visits=1, board=node.state.board
            )

    def play_out(self, state: GameState):
        turns = 0
        while not state.is_terminal():
            turns += 1
            if self.early_term and turns >= self.early_term_turns:
                # Early termination condition
                # ! This assumes symmetric evaluation functions centered around 0!
                evaluation = self.evaluate(state, state.player, norm=True)
                if evaluation > 0.001:
                    return (1, 0) if state.player == 1 else (0, 1)
                elif evaluation < -0.001:
                    return (0, 1) if state.player == 1 else (1, 0)
                else:
                    return (0.5, 0.5)

            if self.dyn_early_term and turns % 5 == 0:
                # Dynamic Early termination condition
                # ! This assumes symmetric evaluation functions centered around 0!
                evaluation = self.evaluate(state, state.player, norm=True)
                if evaluation > self.dyn_early_term_cutoff:
                    return (1, 0) if state.player == 1 else (0, 1)
                elif evaluation < -self.dyn_early_term_cutoff:
                    return (0, 1) if state.player == 1 else (1, 0)

            best_action = None
            # With probability epsilon choose the best action from a subset of moves
            if self.e_greedy and np.random.uniform(0, 1) < self.e_g_epsilon:
                # This presupposes that yield_legal_actions generates moves in a random order
                actions = itertools.islice(state.yield_legal_actions(), self.e_g_subset)
                # * No normalization needed here since it is just used to order the moves
                best_action = max(
                    actions, key=lambda a: self.evaluate(state.apply_action(a), state.player), default=None
                )

            # With probability 1-epsilon chose a move at random
            if best_action is None:
                best_action = state.get_random_action()

            state = state.apply_action(best_action)

        # Map the result to the correct player
        result = state.get_reward(1)
        if result == win:
            result = (1, 0)
        elif result == loss:
            result = (0, 1)
        else:
            result = (0.5, 0.5)
        return result

    def print_cumulative_statistics(self) -> str:
        pass
