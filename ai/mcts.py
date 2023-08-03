# cython: language_level=3

import itertools
import random
from typing import Callable

import cython

from ai.transpos_table import TranspositionTableMCTS
from games.gamestate import loss, win

from cython.cimports.ai.c_random import c_uniform_random, c_random, c_random_seed
from cython.cimports.ai.transpos_table import TranspositionTableMCTS
from cython.cimports.libc.time import time
from cython.cimports.libc.math import sqrt, log

from util import abbreviate, format_time, pretty_print_dict

DEBUG: cython.bint = 0

# TODO Compiler directives toevoegen na debuggen
# TODO After testing, remove assert statements


@cython.cfunc
def curr_time() -> cython.long:
    return time(cython.NULL)


c_random_seed(curr_time())


@cython.cclass
class Node:
    children = cython.declare(cython.list, visibility="public")
    state_hash = cython.declare(cython.longlong, visibility="public")
    im_value: cython.double

    action: cython.tuple
    player: cython.int

    expanded: cython.bint
    solved: cython.bint

    children: cython.list
    actions: cython.list

    tt: TranspositionTableMCTS

    def __init__(
        self,
        player: int,
        state: cython.object,
        action: tuple,
        state_hash: int,
        tt: TranspositionTableMCTS,
    ):
        self.children = []
        # The action that led to this state, needed for root selection
        self.action = action
        self.tt: TranspositionTableMCTS = tt
        self.player = player
        self.state_hash = state_hash

        self.actions = None
        # Since evaluations are stored in view of player 1, player 1 is the maximizing player
        self.im_value = -99999999.9 if self.player == 1 else 99999999.9
        self.expanded = 0
        self.solved = 0
        # Check if the node was previously expanded, in which case add the relevant states to this node.
        self.check_tt_expand(state)

    @cython.ccall
    @cython.locals(
        stats=tuple[cython.double, cython.double, cython.double, cython.int, cython.bint, cython.double]
    )
    def check_tt_expand(self, state: cython.object):
        """
        Check if the node has been expanded elsewhere in the tree since the last time that we saw it
        """
        stats = self.stats()
        # No need to expand a proven node
        if stats[4] == 1 and stats[3] == 0:
            return self.add_all_children(state)

    @cython.cfunc
    @cython.locals(
        my_visist=cython.int,
        max_val=cython.double,
        max_child=Node,
        child=Node,
        child_stats=tuple[
            cython.double, cython.double, cython.double, cython.int, cython.bint, cython.double
        ],
        uct_val=cython.double,
        i=cython.int,
        ci=cython.int,
        n_children=cython.int,
        c=cython.double,
        children_lost=cython.int,
        pb_weight=cython.double,
        imm_alpha=cython.double,
        avg_value=cython.double,
    )
    def uct(self, c, pb_weight=0.0, imm_alpha=0.0) -> Node:
        n_children = len(self.children)

        assert n_children > 0, "Trying to uct a node without children"

        _, _, my_visits, _, _, _ = self.stats()

        # Just to make sure that there's always a child to select
        max_child = self.children[c_random(0, n_children - 1)]

        max_val = -999999.99
        children_lost = 0

        for ci in range(n_children):
            child: Node = self.children[ci]
            # 0: v1, 1: v2, 2: visits, 3: solved_player, 4: is_expanded, 5: eval_value
            child_stats = child.stats()
            if child_stats[3] != 0:
                if child_stats[3] == self.player:  # Winning move, let's go champ
                    # Just in case that the child was solved elsewhere in the tree, update this node.
                    self.set_solved(self.player)
                    return child  # Return the win!
                elif child_stats[3] == 3 - self.player:  # Losing move
                    children_lost += 1
                    continue  # Skip this child, we need to check if all children are losses to mark this node as solved

            avg_value = child_stats[self.player - 1] / child_stats[2]
            # Implicit minimax
            if imm_alpha > 0.0:
                if self.player == 1:
                    avg_value = ((1.0 - imm_alpha) * avg_value) + (imm_alpha * child.im_value)
                if self.player == 2:
                    avg_value = ((1.0 - imm_alpha) * avg_value) + (imm_alpha * -child.im_value)

            # Evaluation is always in view of player 1, so we need to negate the evaluation if the player is 2
            pb_h: cython.double
            if self.player == 2:
                pb_h = -child_stats[5]
            else:
                pb_h = child_stats[5]

            uct_val = (
                avg_value
                + sqrt(c * log(my_visits) / child_stats[2])
                + c_uniform_random(0.0001, 0.001)
                # Progressive bias, set pb_weight to 0 to disable
                + (pb_weight * (pb_h / (1.0 + child_stats[2])))
            )

            if uct_val >= max_val:
                max_child = child
                max_val = uct_val

        # It may happen that somewhere else in the tree all my children have been found to be proven losses.
        if children_lost == n_children:
            # Since all children are losses, we can mark this node as solved for the opponent
            # We do this here because the node may have been expanded and solved elsewhere in the tree
            self.set_solved(3 - self.player)

        return max_child  # A random child was chosen at the beginning of the method, hence this will never be None

    @cython.cfunc
    @cython.locals(
        child=Node,
        action=cython.tuple,
        all_children_loss=cython.bint,
        state=cython.object,
        new_state=cython.object,
        result=cython.int,
        eval_value=cython.double,
        child_stats=tuple[
            cython.double, cython.double, cython.double, cython.int, cython.bint, cython.double
        ],
        my_stats=tuple[cython.double, cython.double, cython.double, cython.int, cython.bint, cython.double],
        prog_bias=cython.bint,
        imm=cython.bint,
        evaluate=cython.object,
    )
    def expand(self, init_state, prog_bias=0, imm=0, evaluate=None) -> Node:
        assert not self.expanded, "Trying to re-expand an already expanded node, you madman."

        if self.actions is None:
            # * Idee: move ordering!
            self.actions = init_state.get_legal_actions()

        while len(self.actions) > 0:
            # Get a random action from the list of previously generated actions
            action = self.actions.pop(c_random(0, len(self.actions) - 1))

            new_state = init_state.apply_action(action)
            child = Node(new_state.player, new_state, action, new_state.board_hash, tt=self.tt)
            self.children.append(child)

            child_stats = child.stats()
            # Solver mechanics..
            if new_state.is_terminal():
                result = new_state.get_reward(self.player)

                if result == win:  # We've found a winning position, hence this node is solved
                    self.set_solved(self.player)
                # We've found a losing position, we cannot yet tell if it is solved, but no need to return the child
                elif result == loss:
                    continue

                # If the game is a draw or a win, we can return the child as any other.
                return child  # Return a child that is not a loss. This is the node we will explore next.

            # This node can also become solved elsewhere in the tree
            elif not self.solved and child_stats[3] != 0:
                if child_stats[3] == self.player:  # This child is a win in the transposition table
                    self.set_solved(self.player)
                    return child  # Return the child, this is the node we will explore next.

                elif child_stats[3] == 3 - self.player:  # The child is a loss in the transposition table
                    continue  # Skip this child, i.e. if all children are losses then we leave the loop with this variable set to 1
            # No use to evaluate terminal or solved nodes.
            elif prog_bias or imm:
                # ! eval_value is always in view of p1
                eval_value = evaluate(new_state, 1, norm=True)
                # Write the evaluation value to the transposition table
                self.tt.put(
                    key=child.state_hash,
                    v1=0,
                    v2=0,
                    visits=0,
                    solved_player=0,
                    is_expanded=0,
                    eval_value=eval_value,
                )
                if imm:
                    child.im_value = eval_value
                    # This means that player 2 is minimizing the evaluated value and player 1 is maximizing it
                    if (self.player == 2 and eval_value < self.im_value) or (
                        self.player == 1 and eval_value > self.im_value
                    ):
                        # The child improves the current evaluation, hence we can update the evaluation value
                        # TODO Volgens mij is dit waar mark het over had, hier was iets mee
                        # TODO Je moet de evaluatie van de parent alleen bijwerken als die verbeterd wordt
                        # TODO Pas als je node helemaal expanded is dan moet je naar alle kids kijken en dan de imValue updaten
                        self.im_value = eval_value
                        # * Idee, blijf nodes toevoegen tot dat je hier bent.
                        # * Dan heb je een node die de imValue verbeterd.
                        # * Je kan zelfs Ni dieper zoeken totdate je de imValue verbeterd hebt en dan pas een node returnen

            return child  # Return the chlid, this is the node we will explore next.

        # If we've reached this point, then the node is fully expanded so we can switch to UCT selection
        self.set_expanded()
        # TODO Remove this after testing
        assert self.check_expanded_node(init_state)
        # Check if all my nodes lead to a loss.
        self.check_loss_node()

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
    def add_all_children(self, init_state: cython.object):
        """
        Adds a previously expanded node's children to the tree
        """
        assert len(self.children) == 0, "add_all_children should only be called on a node without children"

        actions = init_state.get_legal_actions()
        n_actions = len(actions)
        self.expanded = 1
        for a in range(n_actions):
            action = actions[a]
            new_state = init_state.apply_action(action)
            child = Node(new_state.player, new_state, action, new_state.board_hash, tt=self.tt)
            self.children.append(child)

            # The node leads to a win! We can mark this node as solved.
            if child.stats()[3] == self.player:
                self.set_solved(self.player)
                return

        # Check if all children are losses
        self.check_loss_node()

    @cython.cfunc
    @cython.locals(all_children_loss=cython.bint, i=cython.int, opponent=cython.int)
    def check_loss_node(self):
        # I'm solved, nothing to see here.
        if self.stats()[3] != 0:
            return

        all_children_loss = 0
        opponent = 3 - self.player
        child: Node
        for i in range(len(self.children)):
            child = self.children[i]
            # If all children are a loss, then we mark this node as solved for the opponent
            if child.stats()[3] == opponent:
                all_children_loss = 1
            else:
                all_children_loss = 0
                break

        if all_children_loss:  # All children are losses, this node is solved for the opponent
            self.set_solved(opponent)

    @cython.cfunc
    def stats(
        self,
    ) -> tuple[cython.double, cython.double, cython.double, cython.int, cython.bint, cython.double]:
        """
        returns: 0: v1, 1: v2, 2: visits, 3: solved_player, 4: is_expanded, 5: eval_value
        """
        return self.tt.get(self.state_hash)

    @cython.cfunc
    def set_solved(self, solved_player: cython.int):
        """
        Method used to make code more readable.
        Set the given node to solved.

        Args:
            solved_player (cython.int): The player that wins the node
        """
        # Set the node as solved in the tree
        self.solved = 1
        # Set the node as solved in the transposition table
        self.tt.put(
            key=self.state_hash,
            v1=0,
            v2=0,
            visits=0,
            is_expanded=1,
            solved_player=solved_player,
            eval_value=0,
        )

    @cython.cfunc
    def set_expanded(self):
        """
        Method used to make code more readable.
        Set the given node to solved.

        Args:
            solved_player (cython.int): The player that wins the node
        """
        # Mark the node as expanded in the tree
        self.expanded = 1
        # Mark the node as expanded in the transposition table
        self.tt.put(
            key=self.state_hash,
            v1=0,
            v2=0,
            visits=0,
            is_expanded=1,
            solved_player=0,
            eval_value=0,
        )

    @cython.ccall
    @cython.locals(child=Node)
    def check_expanded_node(self, state: cython.object):
        # This means that all children should have been visited.
        assert all([child.stats()[2] > 0 for child in self.children]), (
            f"Node {self} has unvisited children!\n ------ node.children ------ \n"
            + "\n"
            + state.visualize()
            + "\n".join([str(child) for child in self.children])
        )
        assert all([self.tt.exists(child.state_hash) for child in self.children]), (
            "The state is not in the transposition table but should be because its parent is expanded."
            + "\n"
            + state.visualize()
        )
        all_actions = [child.action for child in self.children]
        assert len(all_actions) == len(
            set(all_actions)
        ), f"Non-unique actions in node {self} children: " + str(all_actions)

        return True

    def __str__(self):
        if self.action == ():
            return abbreviate(
                f"Root(P:{self.player}, Hash:{self.state_hash}, Children:{len(self.children)}, Stats:{str(self.stats())})"
            )
        return abbreviate(
            f"Node(P:{self.player}, Action:{self.action}, Hash:{self.state_hash}, Children:{len(self.children)}, Expanded:{self.expanded}, Stats:{str(self.stats())})"
        )


@cython.cclass
class MCTSPlayer:
    player: cython.int
    num_simulations: cython.int
    e_g_subset: cython.int
    transposition_table_size: cython.long
    early_term: cython.bint
    dyn_early_term: cython.bint
    early_term_turns: cython.bint
    early_term_cutoff: cython.double
    e_greedy: cython.bint
    node_priors: cython.bint
    imm: cython.bint
    imm_alpha: cython.double
    debug: cython.bint
    roulette: cython.bint
    dyn_early_term_cutoff: cython.double
    c: cython.double
    epsilon: cython.double
    max_time: cython.double

    evaluate: cython.object
    tt: TranspositionTableMCTS
    root: Node

    def __init__(
        self,
        player: int,
        evaluate: Callable,
        transposition_table_size: int = 2**16,
        num_simulations: int = 0,
        max_time: int = 0,
        c: float = 1.0,
        dyn_early_term: bool = False,
        dyn_early_term_cutoff: float = 0.9,
        early_term: bool = False,
        early_term_turns: int = 10,
        early_term_cutoff: float = 0.05,
        e_greedy: bool = False,
        e_g_subset: int = 20,
        imm_alpha: float = 0.4,
        imm: bool = False,
        roulette: bool = False,
        epsilon: float = 0.05,
        node_priors: bool = False,
        debug: bool = False,
    ):
        # TODO Hier was je gebleven, je moet nog node priors implementeren.

        self.player = player
        self.evaluate: Callable = evaluate

        self.dyn_early_term = dyn_early_term
        self.dyn_early_term_cutoff = dyn_early_term_cutoff
        self.early_term_cutoff = early_term_cutoff
        self.e_greedy = e_greedy
        self.epsilon = epsilon
        self.imm = imm
        self.imm_alpha = imm_alpha
        self.e_g_subset = e_g_subset
        self.early_term = early_term
        self.early_term_turns = early_term_turns
        self.roulette = roulette
        self.c = c

        self.node_priors = node_priors
        # either we base the time on a fixed number of simulations or on a fixed time
        if num_simulations:
            self.num_simulations = num_simulations
            self.max_time = 0
        elif max_time:
            self.max_time = max_time
            self.num_simulations = 0
        else:
            assert (
                self.max_time is not None or num_simulations is not None
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
        max_value=cython.double,
        n_children=cython.int,
        node=Node,
        value=cython.double,
        start_time=cython.long,
        i=cython.int,
    )
    def best_action(self, state: cython.object):
        # Reset root for new round of MCTS
        self.root = Node(state.player, state, (), state.board_hash, self.tt)

        start_time = curr_time()
        i = 0
        if self.num_simulations:
            for i in range(self.num_simulations):
                if DEBUG:
                    if i + 1 % 100 == 0:
                        print(f"\rSimulation: {i+1} ", end="")
                self.simulate(state)
        else:
            while curr_time() - start_time < self.max_time:
                if DEBUG:
                    if i % 100 == 0:
                        print(f"\rSimulation: {i+1} ", end="")
                self.simulate(state)
                i += 1

        if DEBUG:
            total_time: cython.long = curr_time() - start_time
            print(
                f"Ran {i+1} simulations in {format_time(total_time)}, {i / (total_time):.1f} simulations per second."
            )

        # retrieve the node with the most visits
        assert len(self.root.children) > 0, "No children found for root node"
        max_node = self.root.children[0]  # TODO Hier gaat het mis (segmentation error)
        max_value = max_node.stats()[2]
        n_children = len(self.root.children)

        for i in range(1, n_children):
            node = self.root.children[i]
            value = node.stats()[2]
            if value > max_value:
                max_node = node
                max_value = value

        if DEBUG:
            print("--*--" * 50)
            print(f"Max node found: {max_node}, with max value: {max_value}")

        if DEBUG:
            print("\n\t".join([str(child) for child in self.root.children]))
            print(f"{self.root}")
            pretty_print_dict(self.tt.get_metrics())

        # Clean the transposition table
        self.tt.evict()

        return max_node.action, max_value  # return the most visited state

    @cython.cfunc
    def simulate(self, init_state: cython.object):
        node: Node = self.root
        selected: cython.list = [self.root]

        next_state: cython.object = init_state

        is_terminal: cython.bint = 0

        # Select: non-terminal, non-solved, expanded nodes
        while not is_terminal and node.expanded and node.stats()[3] == 0:
            # ? The question here is, is it really important that I encounter the odd collision?
            # ? Because avoiding collistions means performance decrease.
            assert (
                node.state_hash == next_state.board_hash
            ), f"Very bad! The state hashes don't match! {node.state_hash} != {next_state.board_hash}"

            node = node.uct(self.c)
            next_state = next_state.apply_action(node.action)
            selected.append(node)

            if node.stats()[3] != 0:  # If the node is solved, we don't need to expand it or simulate it
                break

            is_terminal = next_state.is_terminal()

        result: tuple[cython.double, cython.double]
        # If the node is neither terminal nor solved, then we need a playout
        if not is_terminal and not node.stats()[3]:
            next_node: Node = node.expand(next_state)  # Expansion returns the expanded node

            # This is the point where the last action was previously added to the node, so in fact the node is just marked as expanded
            if next_node == None:
                next_node = node.uct(self.c)

            # place the last move
            next_state = next_state.apply_action(next_node.action)
            selected.append(next_node)

            # Do a random playout and collect the result
            result = self.play_out(next_state)

        else:  # TODO If a proven node is returned, we should backpropagate the result of the state
            p1_reward: cython.double = 0.0
            if is_terminal:
                # A terminal node is reached, so we can backpropagate the result of the state as if it was a playout
                p1_reward = next_state.get_reward(1)
            elif node.stats()[3] == 1:
                p1_reward = win
            elif node.stats()[3] == 2:
                p1_reward = loss
            else:
                assert False, "This should not happen!"

            if p1_reward == win:
                result = (1.0, 0.0)
            elif p1_reward == loss:
                result = (0.0, 1.0)
            else:
                result = (0.5, 0.5)

        i: cython.int
        c: cython.int
        min_im_val: cython.double
        max_im_val: cython.double
        # Backpropagate the result along the chosen nodes
        for i in range(len(selected), 0, -1):  # Move backwards through the list
            node = selected[i - 1]  # In reverse, start is inclusive, stop is exclusive
            if self.imm and node.expanded:  # The last node selected (first in the loop) is not expanded
                # Update the im_values of the node based on min/maxing the im_values of the children
                if node.player == 1:  # maximize im_value
                    max_im_val = -99999999.9
                    for c in range(len(node.children)):
                        max_im_val = max(max_im_val, node.children[c].im_value)
                    node.im_value = max_im_val
                elif node.player == 2:  # minimize im_value
                    min_im_val = 99999999.9
                    for c in range(len(node.children)):
                        min_im_val = min(min_im_val, node.children[c].im_value)
                    node.im_value = min_im_val

            self.tt.put(
                key=node.state_hash,
                v1=result[0],
                v2=result[1],
                visits=1,
                solved_player=0,
                is_expanded=0,
                eval_value=0,
            )

    @cython.ccall
    @cython.returns(tuple[cython.double, cython.double])
    @cython.locals(
        state=cython.object,
        turns=cython.int,
        evaluation=cython.double,
        actions=cython.list,
        best_action=cython.tuple,
        action=cython.tuple,
        reward=cython.int,
        max_value=cython.double,
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
                if evaluation > self.early_term_cutoff:
                    return (1.0, 0.0) if state.player == 1 else (0.0, 1.0)
                elif evaluation < -self.early_term_cutoff:
                    return (0.0, 1.0) if state.player == 1 else (1.0, 0.0)
                else:
                    return (0.5, 0.5)

            # Dynamic Early termination condition, check every 5 turns if the evaluation has a certain value
            if self.dyn_early_term == 1 and turns % 5 == 0:
                # ! This assumes symmetric evaluation functions centered around 0!
                # TODO Figure out the a (max range) for each evaluation function
                evaluation = self.evaluate(state, state.player, norm=True)

                if evaluation > self.dyn_early_term_cutoff:
                    return (1.0, 0.0) if state.player == 1 else (0.0, 1.0)
                elif evaluation < -self.dyn_early_term_cutoff:
                    return (0.0, 1.0) if state.player == 1 else (1.0, 0.0)

            best_action: cython.tuple = ()
            # With probability epsilon choose the best action from a subset of moves
            if self.e_greedy == 1 and c_uniform_random(0, 1) < self.epsilon:
                # This presupposes that yield_legal_actions generates moves in a random order
                actions = list(itertools.islice(state.yield_legal_actions(), self.e_g_subset))
                max_value = -99999.99
                for action in actions:
                    value = self.evaluate(state.apply_action(action), state.player)
                    if value > max_value:
                        max_value = value
                        best_action = action

            # With probability epsilon choose a move using roulette wheel selection based on the move ordering
            if self.roulette == 1 and c_uniform_random(0, 1) < self.epsilon:
                actions = state.get_legal_actions()
                best_action = random.choices(actions, weights=state.move_weights(actions), k=1)[0]

            # With probability 1-epsilon chose a move at random
            if best_action == ():
                best_action = state.get_random_action()

            state = state.apply_action(best_action)

        # Map the result to the players
        reward = state.get_reward(1)

        if reward == win:
            return (1.0, 0.0)
        elif reward == loss:
            return (0.0, 1.0)
        else:
            return (0.5, 0.5)

    def print_cumulative_statistics(self) -> str:
        return ""

    def __repr__(self):
        # Try to get the name of the evaluation function, which can be a partial
        try:
            eval_name = self.evaluate.__name__
        except AttributeError:
            eval_name = self.evaluate.func.__name__

        return abbreviate(
            f"MCTS(p={self.player}, evaluate={eval_name}, "
            f"num_simulations={self.num_simulations}, "
            f"max_time={self.max_time}, c={self.c}, dyn_early_term={self.dyn_early_term}, "
            f"dyn_early_term_cutoff={self.dyn_early_term_cutoff}, early_term={self.early_term}, "
            f"early_term_turns={self.early_term_turns}, e_greedy={self.e_greedy}, "
            f"e_g_subset={self.e_g_subset}, roulette={self.roulette}, epsilon={self.epsilon}, "
            f"node_priors={self.node_priors})"
        )
