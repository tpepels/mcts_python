# cython: language_level=3

import random
import cython

import itertools
import time

import numpy as np
from ai.transpos_table import TranspositionTableMCTS
from games.gamestate import win, loss
from cython.cimports.ai.transpos_table import TranspositionTableMCTS

from util import abbreviate, pretty_print_dict

DEBUG: cython.bint = 0


@cython.cclass
class Node:
    children = cython.declare(cython.list, visibility="public")

    action: cython.tuple
    state_hash: cython.longlong
    player: cython.int
    expanded: cython.bint
    action_generator: cython.object
    children: cython.list

    tt: TranspositionTableMCTS

    def __init__(
        self, player: int, state: cython.object, action: tuple, state_hash: int, tt: TranspositionTableMCTS
    ):
        self.children = []
        # The action that led to this state, needed for root selection
        self.action = action
        self.tt = tt
        self.player = player
        self.state_hash = state_hash
        self.action_generator = None
        self.expanded = 0

        # Check if the node was previously expanded, in which case add the relevant states to this node.
        self.check_tt_expand(state)

    @cython.ccall
    @cython.locals(stats=tuple[cython.float, cython.float, cython.float, cython.int, cython.int, cython.bint])
    def check_tt_expand(self, state: cython.object):
        """
        Check if the node has been expanded elsewhere in the tree since the last time that we saw it
        """
        stats = self.stats()
        # No need to expand an already expanded node, since all nodes are already in the tt
        if not self.expanded and stats[5] and stats[4] == 0:
            return self.add_all_children(state)

    @cython.cfunc
    @cython.locals(
        my_visist=cython.int,
        max_val=cython.float,
        max_child=Node,
        child=Node,
        child_stats=tuple[cython.float, cython.float, cython.float, cython.int, cython.int, cython.bint],
        uct_val=cython.float,
        i=cython.int,
        ci=cython.int,
        n_children=cython.int,
        c=cython.float,
        children_lost=cython.int,
    )
    @cython.returns(Node)
    def uct(self, c) -> Node:
        _, _, _, my_visits, _, _ = self.stats()

        max_val = -999999.99
        max_child = None
        n_children = len(self.children)
        children_lost = 0
        for ci in range(n_children):
            child = self.children[ci]
            # 0: v1, 1: v2, 2: im_value, 3: visits, 4: solved_player, 5: is_expanded
            child_stats = child.stats()
            # TODO Hier was je gebleven, je kan dus nodes hebben die zonder visits zitten omdat die ergens anders aan de tree toegevoegd zijn
            # TODO Maar dat moet je wel nog even bevestigen.
            assert (
                child_stats[3] > 0 or child_stats[4] != 0
            ), f"Child ({child}) of mine ({self}) has neither visits in uct, nor is it solved!"

            i = self.player - 1  # This makes is easier to index the player in the stats

            # TODO Dit moet je nog checken
            if child_stats[4] == self.player:  # Winning move, let's go champ
                self.tt.put(
                    key=self.state_hash,
                    v1=0,
                    v2=0,
                    visits=0,
                    im_value=0,
                    is_expanded=1,
                    solved_player=self.player,
                )
                return child
            elif child_stats[4] == 3 - self.player:  # Losing move
                children_lost += 1
                continue  # Skip this child

            uct_val = (
                child_stats[i] / child_stats[3]
                + np.sqrt(c * np.log(my_visits) / child_stats[3])
                + np.random.uniform(0.0001, 0.001)
            )

            if uct_val >= max_val:
                max_child = child
                max_val = uct_val

        # It may happen that somewhere else in the tree all my children have been found to be proven losses.
        if children_lost == len(self.children):
            self.tt.put(
                key=self.state_hash,
                v1=0,
                v2=0,
                visits=0,
                im_value=0,
                is_expanded=1,
                solved_player=3 - self.player,
            )
            # Just return any child, it doesn't matter which one because they are all losses.
            return random.choice(self.children)

        return max_child

    @cython.cfunc
    @cython.locals(
        child=Node,
        action=cython.tuple,
        all_children_loss=cython.bint,
        state=cython.object,
        new_state=cython.object,
        result=cython.int,
    )
    def expand(self, state) -> Node:
        # If no generator exists, create it
        if self.action_generator is None:
            # TODO Checken of dit betere performance geeft dan get_legal_actions
            self.action_generator = state.yield_legal_actions()

        for action in self.action_generator:
            # TODO Remove this after testing
            assert action not in [child.action for child in self.children]

            new_state = state.apply_action(action)
            child = Node(new_state.player, state, action, new_state.board_hash, tt=self.tt)
            # print(f"Adding node {child} to the tree.")
            self.children.append(child)

            if new_state.is_terminal():  # Solver
                result = new_state.get_reward(self.player)
                if result == win:  # We've found a winning position, hence this node is solved
                    self.tt.put(
                        key=self.state_hash,
                        v1=0,
                        v2=0,
                        visits=0,
                        im_value=0,
                        is_expanded=1,
                        solved_player=self.player,
                    )
                    return child
                # We've found a losing position, we cannot yet tell if it is solved, but no need to return the child
                elif result == loss:
                    continue
                # If the game is a draw, we can just continue and return the child as any other
            elif child.stats()[4] == self.player:  # This child is a win in the transposition table
                self.tt.put(
                    key=self.state_hash,
                    v1=0,
                    v2=0,
                    visits=0,
                    im_value=0,
                    is_expanded=1,
                    solved_player=self.player,
                )
                return child
            elif child.stats()[4] == 3 - self.player:  # The child is a loss in the transposition table
                continue  # Skip this child, i.e. if all children are losses then we leave the loop with this variable set to 1

            return child  # Return a child that is not a loss. This is the node we will explore next.

        # print(f"All children of {self} are added to the tree.")
        # The node is fully expanded so we can switch to UTC selection
        self.expanded = 1
        self.tt.put(
            key=self.state_hash,
            v1=0,
            v2=0,
            visits=0,
            im_value=0,
            is_expanded=1,
            solved_player=0,
        )

        self.check_loss_node()
        # return any child, the node is now expanded so we can switch to UTC selection
        return random.choice(self.children)

    @cython.cfunc
    @cython.locals(all_children_loss=cython.bint, i=cython.int)
    def check_loss_node(self):
        all_children_loss = 0
        for i in range(len(self.children)):
            # If all children are a loss, then we mark this node as solved for the opponent
            if self.children[i].stats()[4] == 3 - self.player:
                all_children_loss = 1
            else:
                all_children_loss = 0
                break

        if all_children_loss:  # All children are losses, this node is solved for the opponent
            self.tt.put(
                key=self.state_hash,
                v1=0,
                v2=0,
                visits=0,
                im_value=0,
                is_expanded=1,
                solved_player=3 - self.player,
            )

    @cython.cfunc
    @cython.locals(
        actions=cython.list,
        child=Node,
        action=cython.tuple,
        new_state=cython.object,
        n_actions=cython.int,
        n_children=cython.int,
        a=cython.int,
        c=cython.int,
    )
    def add_all_children(self, state: cython.object):
        """
        Adds a previously expanded node's children to the tree
        """
        actions = state.get_legal_actions()
        n_actions = len(actions)
        n_children = len(self.children)
        for a in range(n_actions):
            action = actions[a]
            new_state = state.apply_action(action)

            for c in range(n_children):
                child = self.children[c]
                if child.action == action:
                    continue  # Skip this child, it already exists

            child = Node(new_state.player, new_state, action, new_state.board_hash, tt=self.tt)
            self.children.append(child)

        self.expanded = 1
        # Check if all children are losses
        self.check_loss_node()

    @cython.ccall
    def stats(self) -> tuple[cython.float, cython.float, cython.float, cython.int, cython.int, cython.bint]:
        """
        returns: 0: v1, 1: v2, 2: im_value, 3: visits, 4: solved_player, 5: is_expanded
        """
        return self.tt.get(self.state_hash)

    def __str__(self):
        if self.action == ():
            return abbreviate(
                f"Root(P:{self.player}, Hash: {self.state_hash}, Children: {len(self.children)}, Stats: {str(self.stats())})"
            )

        return abbreviate(
            f"Node(P: {self.player}, Action: {self.action}, Hash: {self.state_hash}, Children {len(self.children)}, Expanded: {self.expanded}, Stats: {str(self.stats())})"
        )


@cython.cclass
class MCTSPlayer:
    player: cython.int
    evaluate: cython.object
    transposition_table_size: cython.long
    num_simulations: cython.int
    c: cython.float
    dyn_early_term: cython.bint
    dyn_early_term_cutoff: cython.float
    early_term: cython.bint
    early_term_turns: cython.bint
    e_greedy: cython.bint
    epsilon: cython.float
    e_g_subset: cython.int
    node_priors: cython.bint
    debug: cython.bint
    roulette: cython.bint
    time: cython.float

    tt: TranspositionTableMCTS
    root: Node

    def __init__(
        self,
        player: int,
        evaluate,
        transposition_table_size: int = 2**16,
        num_simulations: int = 0,
        max_time: int = 0,
        c: float = 1.0,
        dyn_early_term: bool = False,
        dyn_early_term_cutoff: float = 0.9,
        early_term: bool = False,
        early_term_turns: int = 10,
        e_greedy: bool = False,
        e_g_subset: int = 20,
        roulette: bool = False,
        epsilon: float = 0.05,
        node_priors: bool = False,
        debug: bool = False,
    ):
        self.player = player
        self.evaluate = evaluate

        self.dyn_early_term = dyn_early_term
        self.dyn_early_term_cutoff = dyn_early_term_cutoff
        self.e_greedy = e_greedy
        self.epsilon = epsilon
        self.e_g_subset = e_g_subset
        self.early_term = early_term
        self.early_term_turns = early_term_turns
        self.roulette = roulette
        self.c = c

        self.node_priors = node_priors
        # either we base the time on a fixed number of simulations or on a fixed time
        if num_simulations:
            self.num_simulations = num_simulations
            self.time = 0
        elif max_time:
            self.time = max_time
            self.num_simulations = 0
        else:
            assert (
                time is not None or num_simulations is not None
            ), "Either provide num_simulations or search time"

        # Because more than one class is involved, just set a global flag
        if debug:
            global DEBUG
            DEBUG = 1

        self.tt = TranspositionTableMCTS(transposition_table_size)

    @cython.ccall
    @cython.returns(cython.tuple)
    @cython.locals(
        state=cython.object,
        max_node=Node,
        max_value=cython.float,
        n_children=cython.int,
        node=Node,
        value=cython.float,
        start_time=cython.float,
    )
    def best_action(self, state: cython.object):
        # Reset root for new round of MCTS
        self.root = Node(state.player, state, (), state.board_hash, self.tt)

        if self.num_simulations:
            if DEBUG:
                print(f"Running simulations. Number of simulations: {self.num_simulations}")
            for _ in range(self.num_simulations):
                self.simulate(state)
        else:
            start_time = time.time()
            if DEBUG:
                print(f"Running simulations based on time limit: {self.time}")
            while time.time() - start_time < self.time:
                self.simulate(state)

        # retrieve the node with the most visits
        max_node = self.root.children[0]
        max_value = max_node.stats()[3]
        n_children = len(self.root.children)
        for i in range(1, n_children):
            node = self.root.children[i]
            value = node.stats()[3]
            if value > max_value:
                max_node = node
                max_value = value

        if DEBUG:
            print(f"Max node found: {max_node}, with max value: {max_value}")

        if DEBUG:
            print("\n\t".join([str(child) for child in self.root.children]))
            print(f"{self.root}")
            pretty_print_dict(self.tt.get_metrics())

        # Clean the transposition table
        self.tt.evict()

        return max_node.action, max_value  # return the most visited state

    @cython.cfunc
    def simulate(self, state: cython.object):
        node: Node = self.root
        selected: cython.list = [node.state_hash]
        next_state: cython.object = state
        result: tuple[cython.float, cython.float]

        while not next_state.is_terminal() and node.expanded:
            node = node.uct(self.c)
            next_state = state.apply_action(node.action)
            selected.append(node.state_hash)

            if node.stats()[4] != 0:  # If the node is solved, we don't need to expand it or simulate it
                break

        if not next_state.is_terminal():
            node = node.expand(next_state)
            next_state = state.apply_action(node.action)
            selected.append(node.state_hash)

            # Do a random playout and collect the result
            result = self.play_out(next_state)
            # if DEBUG:
            #     print(f"Random playout result: {result}")
        else:
            p1_reward: cython.float = next_state.get_reward(1)
            if p1_reward == win:
                result = (1.0, 0.0)
            elif p1_reward == loss:
                result = (0.0, 1.0)
            else:
                result = (0.5, 0.5)
            # if DEBUG:
            #     print(f"Terminal node result: {result}")

        # TODO If a proven node is returned, we should not backpropagate the result
        # TODO Hier was je gebleven, backprop doet niks, effe tt.put checken
        # TODO Tictactoe get_reward klopte niet, dus die experimenten moeten opnieuw.
        n_sel: cython.int = len(selected)
        i: cython.int
        # Backpropagate the result along the chosen nodes
        for i in range(n_sel):
            self.tt.put(
                key=selected[i],
                v1=result[0],
                v2=result[1],
                visits=1,
                im_value=0,
                solved_player=0,
                is_expanded=0,
            )

    @cython.ccall
    @cython.returns(tuple[cython.float, cython.float])
    @cython.locals(
        state=cython.object,
        turns=cython.int,
        evaluation=cython.float,
        actions=cython.list,
        best_action=cython.tuple,
        action=cython.tuple,
        result=tuple[cython.float, cython.float],
        max_value=cython.float,
    )
    def play_out(self, state: cython.object):
        turns = 0
        while not state.is_terminal():
            turns += 1

            # Early termination condition with a fixed number of turns
            if self.early_term and turns >= self.early_term_turns:
                # ! This assumes symmetric evaluation functions centered around 0!
                # TODO Figure out the a (max range) for each evaluation function
                evaluation = self.evaluate(state, state.player, norm=True)
                if evaluation > 0.001:
                    return (1, 0) if state.player == 1 else (0, 1)
                elif evaluation < -0.001:
                    return (0, 1) if state.player == 1 else (1, 0)
                else:
                    return (0.5, 0.5)

            # Dynamic Early termination condition, check every 5 turns if the evaluation has a certain value
            if self.dyn_early_term and turns % 5 == 0:
                # ! This assumes symmetric evaluation functions centered around 0!
                # TODO Figure out the a (max range) for each evaluation function
                evaluation = self.evaluate(state, state.player, norm=True)
                if evaluation > self.dyn_early_term_cutoff:
                    return (1, 0) if state.player == 1 else (0, 1)
                elif evaluation < -self.dyn_early_term_cutoff:
                    return (0, 1) if state.player == 1 else (1, 0)

            best_action = ()
            # With probability epsilon choose the best action from a subset of moves
            if self.e_greedy and np.random.uniform(0, 1) < self.epsilon:
                # This presupposes that yield_legal_actions generates moves in a random order
                actions = list(itertools.islice(state.yield_legal_actions(), self.e_g_subset))
                max_value = -99999.99
                for action in actions:
                    value = self.evaluate(state.apply_action(action), state.player)
                    if value > max_value:
                        max_value = value
                        best_action = action

            # With probability epsilon choose a move using roulette wheel selection based on the move ordering
            if self.roulette and np.random.uniform(0, 1) < self.epsilon:
                actions = state.get_legal_actions()
                best_action = random.choices(actions, weights=state.move_weights(actions), k=1)[0]

            # With probability 1-epsilon chose a move at random
            if best_action == ():
                best_action = state.get_random_action()

            state = state.apply_action(best_action)

        # Map the result to the correct player
        reward = state.get_reward(1)
        result: tuple[cython.float, cython.float]

        if reward == win:
            result = (1.0, 0.0)
        elif reward == loss:
            result = (0.0, 1.0)
        else:
            result = (0.5, 0.5)

        return result

    def print_cumulative_statistics(self) -> str:
        pass

    def __repr__(self):
        return f"<MCTS player={self.player}, evaluate={self.evaluate}, transposition_table_size={2**16}, num_simulations={self.num_simulations}, max_time={self.time}, c={self.c}, dyn_early_term={self.dyn_early_term}, dyn_early_term_cutoff={self.dyn_early_term_cutoff}, early_term={self.early_term}, early_term_turns={self.early_term_turns}, e_greedy={self.e_greedy}, e_g_subset={self.e_g_subset}, roulette={self.roulette}, epsilon={self.epsilon}, node_priors={self.node_priors}, debug={self.debug}>"
