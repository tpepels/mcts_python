# cython: language_level=3

import gc
import random
from random import random as rand_float
from typing import Optional
from colorama import Back, Fore, init

from includes.dynamic_bin import DynamicBin

init(autoreset=True)
import cython
from cython.cimports.libc.time import time
from cython.cimports.libc.math import sqrt, log, INFINITY, isnan
from cython.cimports.includes import GameState, win, loss

from util import format_time

DEBUG: cython.bint = __debug__
prunes: cython.int = 0
non_prunes: cython.int = 0
ab_bound: cython.int = 0
ucb_bound: cython.int = 0

if __debug__:
    dynamic_bins: cython.dict = {}
    n_bins: cython.short = 15
    dynamic_bins["alpha"] = {"bin": DynamicBin(n_bins), "label": "alpha values"}
    dynamic_bins["beta"] = {"bin": DynamicBin(n_bins), "label": "beta values"}
    dynamic_bins["alpha_bounds"] = {"bin": DynamicBin(n_bins), "label": "alpha bounds"}
    dynamic_bins["beta_bounds"] = {"bin": DynamicBin(n_bins), "label": "beta bounds"}
    dynamic_bins["k"] = {"bin": DynamicBin(n_bins), "label": "k"}
    dynamic_bins["k_comp"] = {"bin": DynamicBin(n_bins), "label": "k_factor * sqrt(log((1 + k) * p_n))"}


@cython.cfunc
@cython.inline
@cython.exceptval(-99999999.9, check=False)
def uniform(a: cython.float, b: cython.float) -> cython.float:
    "Get a random number in the range [a, b) or [a, b] depending on rounding."
    mod: cython.float = a + (b - a)
    return mod * rand_float()


@cython.cfunc
@cython.exceptval(-1, check=False)
def curr_time() -> cython.long:
    return time(cython.NULL)


@cython.cclass
class Node:
    children = cython.declare(cython.list[Node], visibility="public")
    # State values
    v = cython.declare(cython.float[2], visibility="public")
    n_visits = cython.declare(cython.int, visibility="public")
    expanded = cython.declare(cython.bint, visibility="public")
    im_value = cython.declare(cython.float, visibility="public")
    solved_player = cython.declare(cython.short, visibility="public")
    player = cython.declare(cython.short, visibility="public")
    anti_decisive = cython.declare(cython.bint, visibility="public")
    # Private fields
    draw: cython.bint
    eval_value: cython.float
    action: cython.tuple
    actions: cython.list  # A list of actions that is used to expand the node

    def __cinit__(self, player: cython.short, action: cython.tuple, max_player: cython.short):
        # The action that led to this state, needed for root selection
        self.action = action
        self.player = player
        # Now we know whether we are minimizing or maximizing
        self.im_value = -INFINITY if self.player == max_player else INFINITY
        self.eval_value = 0.0
        self.expanded = 0
        self.solved_player = 0
        self.v = [0.0, 0.0]  # Values for player-1
        self.n_visits = 0
        self.actions = []
        self.children = []

    @cython.cfunc
    def uct(
        self,
        c: cython.float,
        max_player: cython.short,
        pb_weight: cython.float = 0.0,
        imm_alpha: cython.float = 0.0,
        ab_p1: cython.short = 0,
        ab_p2: cython.short = 0,
        alpha: cython.float = -INFINITY,
        beta: cython.float = INFINITY,
        k_factor: cython.float = 1.0,
        alpha_bounds: cython.float = 0.0,
        beta_bounds: cython.float = 0.0,
    ) -> Node:
        n_children: cython.Py_ssize_t = len(self.children)
        assert self.expanded, "Trying to uct a node that is not expanded"
        global ucb_bound, ab_bound

        selected_child: Optional[Node] = None
        best_val: cython.double = -INFINITY
        children_lost: cython.short = 0
        children_draw: cython.short = 0

        p_n: cython.float = cython.cast(cython.float, max(1, self.n_visits))

        if ab_p1 == 2 and alpha != -INFINITY and beta != INFINITY:
            # Here alpha can be bigger than beta. Beta_bounds is always positive, alpha_bounds is always negative
            if ab_p2 == 1 or ab_p2 == 3 or ab_p2 == 5:
                k: cython.float = (beta - alpha) * (1 - (beta_bounds - alpha_bounds))
            elif ab_p2 == 2 or ab_p2 == 4 or ab_p2 == 6:
                k: cython.float = beta - alpha

            k *= k_factor

            if k != 0:
                # This is the case where the bounds are used to adjust the UCT value
                if ab_p2 == 1 or ab_p2 == 2:
                    c *= 1 + k

                    if __debug__:  # Add the value to the dynamic bin
                        dynamic_bins["k_comp"].get("bin").add_data(1 + k)

                elif ab_p2 == 3 or ab_p2 == 4:
                    c *= 1 + log(1 + k)

                    if __debug__:  # Add the value to the dynamic bin
                        dynamic_bins["k_comp"].get("bin").add_data(1 + log(1 + k))

                elif ab_p2 == 5 or ab_p2 == 6:
                    c *= sqrt(1 + log(1 + k))

                    if __debug__:  # Add the value to the dynamic bin
                        dynamic_bins["k_comp"].get("bin").add_data(sqrt(1 + log(1 + k)))

                ab_bound += 1
            else:
                ucb_bound += 1
        else:
            ucb_bound += 1

        # Move through the children to find the one with the highest UCT value
        ci: cython.short
        for ci in range(n_children):
            child: Node = self.children[ci]
            # ! A none exception could happen in case there's a mistake in how the children are added to the list in expand
            # Make sure that every child is seen at least once.
            if child.n_visits == 0:
                return child

            c_n: cython.float = cython.cast(cython.float, child.n_visits)

            # Check for solved children
            if child.solved_player != 0:
                if child.solved_player == self.player:  # Winning move, let's go champ
                    # In case that the child was solved deeper in the tree, update this node.
                    self.solved_player = self.player
                    return child  # Return the win!
                elif child.solved_player == 3 - self.player:  # Losing move
                    children_lost += 1
                    continue  # Skip this child, we need to check if all children are losses to mark this node as solved

            # Check whether all moves lead to a draw
            if child.draw:
                children_draw += 1

            uct_val: cython.double = child.get_value_imm(self.player, imm_alpha, max_player) + (
                c * sqrt(log(p_n) / c_n)
            )

            if ab_p1 == 3 and alpha != -INFINITY and beta != INFINITY:
                # Change the UCT value based on the bounds
                if uct_val >= alpha and uct_val <= beta:
                    uct_val += beta
                elif uct_val < alpha:
                    uct_val += alpha_bounds
                elif uct_val > beta:
                    uct_val += -beta_bounds

                ab_bound += 1

            if pb_weight > 0.0:
                uct_val += pb_weight * (cython.cast(cython.float, child.eval_value) / (1.0 + c_n))

            assert not isnan(uct_val), f"UCT value is NaN!.\nNode: {str(self)}\nChild: {str(child)}"

            rand_fact: cython.float = uniform(-0.0001, 0.0001)

            # Find the highest UCT value
            if (uct_val + rand_fact) >= best_val:
                selected_child = child
                best_val = uct_val

        # It may happen that somewhere else in the tree all my children have been found to be proven losses.
        if children_lost == n_children:
            # Since all children are losses, we can mark this node as solved for the opponent
            # We do this here because the node may have been expanded and solved elsewhere in the tree
            self.solved_player = 3 - self.player
            # just return a random child, they all lead to a loss anyway
            return random.choice(self.children)
        elif children_lost == (n_children - 1):
            self.check_loss_node()

            if self.solved_player == 0 or self.solved_player == self.player:
                # There's only one move that does not lead to a loss. This is an anti-decisive move.
                self.anti_decisive = 1

        # Proven draw
        elif children_draw == n_children:
            self.draw = 1

        assert selected_child is not None, f"No child selected in UCT! {str(self)}"

        return selected_child

    @cython.cfunc
    def expand(
        self,
        init_state: GameState,
        max_player: cython.short,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
    ) -> Node:
        assert not self.expanded, f"Trying to re-expand an already expanded node, you madman. {str(self)}"
        assert self.player == init_state.player, f"Player mismatch in expand! {self.player=} != {init_state.player=}"

        if self.children == []:
            # * Idee: move ordering!
            self.actions = init_state.get_legal_actions()
            # in MiniShogi we reach positions with None pass moves, which mean the game is over.
            if self.actions[0] == None and len(self.actions) == 1:

                assert init_state.is_terminal(), "State with None action is not terminal.."

                self.actions = []
                # A pass move means that I lost because I have no remaining moves
                self.solved_player = 3 - self.player
                self.im_value = loss

            self.children = [None] * len(self.actions)

        while len(self.actions) > 0:
            # Get a random action from the list of previously generated actions
            action: cython.tuple = self.actions.pop()

            new_state: GameState = init_state.apply_action(action)
            child: Node = Node(new_state.player, action, max_player)

            # This works because we pop the actions
            self.children[len(self.children) - len(self.actions) - 1] = child

            # Solver mechanics..
            if new_state.is_terminal():
                result: cython.int = new_state.get_reward(self.player)

                if result == win:  # We've found a winning position, hence this node and the child are solved
                    self.solved_player = child.solved_player = self.player
                    child.im_value = win
                # We've found a losing position, we cannot yet tell if it is solved, but no need to return the child
                elif result == loss:
                    child.im_value = loss
                    child.solved_player = 3 - self.player
                    continue
                else:
                    child.im_value = 0.0
                    child.draw = 1
                # If the game is a draw or a win, we can return the child as any other.
                return child  # Return a child that is not a loss. This is the node we will explore next.
            # No use to evaluate terminal or solved nodes.
            elif prog_bias or imm:
                # set the evaluation value to the evaluation of the state
                if prog_bias:
                    # * eval_value is always in view of max_player
                    eval_value: cython.float = new_state.evaluate(
                        player=max_player,
                        norm=True,
                        params=eval_params,
                    ) + uniform(-0.0001, 0.0001)
                    child.eval_value = eval_value

                if imm:
                    # At first, the im value of the child is the same as the evaluation value, when searching deeper, it becomes the min/max of the subtree
                    child.im_value = new_state.evaluate(
                        player=max_player,
                        norm=True,
                        params=eval_params,
                    ) + uniform(-0.00001, 0.00001)
                    child.n_visits += 1

                    if (self.player != max_player and child.im_value < self.im_value) or (
                        self.player == max_player and child.im_value > self.im_value
                    ):
                        # The child improves the current evaluation, hence we can update the evaluation value
                        self.im_value = child.im_value

            return child  # Return the chlid, this is the node we will explore next.

        # If we've reached this point, then the node is fully expanded so we can switch to UCT selection
        self.expanded = 1

        # We reached a node with no successors, this means loss/draw depending on the game
        if len(self.children) == 0:
            assert (
                self.solved_player != 0
            ), f"Node {str(self)} has no children, but is not solved {init_state.visualize(True)}"
            return None

        # Check if all my nodes lead to a loss.
        self.check_loss_node()

        if imm:
            # Do the full minimax back-up
            best_im: cython.float
            best_node: Node

            best_im = -INFINITY if self.player == max_player else INFINITY

            i: cython.short
            for i in range(len(self.children)):
                child = self.children[i]

                # * Minimize or Maximize my im value over all children
                if (self.player == max_player and child.im_value > best_im) or (
                    self.player != max_player and child.im_value < best_im
                ):
                    best_im = child.im_value
                    best_node = child

            self.im_value = best_im
            # Return the best node for another visit
            return best_node

        # Return a random child for another visit
        return random.choice(self.children)

    @cython.cfunc
    def add_all_children(
        self,
        init_state: GameState,
        max_player: cython.short,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
    ):
        """
        Adds a previously expanded node's children to the tree
        """
        while not self.expanded:
            self.expand(init_state, max_player=max_player, prog_bias=prog_bias, imm=imm, eval_params=eval_params)

    @cython.cfunc
    @cython.locals(child=Node, i=cython.short)
    def check_loss_node(self):
        # I'm solved, nothing to see here.
        if self.solved_player != 0:
            return

        for i in range(len(self.children)):
            child = self.children[i]
            # If all children lead to a loss, then we will not return from the function
            if child.solved_player != 3 - self.player:
                return

        self.solved_player = 3 - self.player

    @cython.cfunc
    @cython.inline
    def get_value_with_uct_interval(
        self,
        c: cython.float,
        player: cython.short,
        max_player: cython.short,
        imm_alpha: cython.float,
        N: cython.int,
    ) -> cython.tuple[cython.float, cython.float]:
        value: cython.float = self.get_value_imm(player, imm_alpha, max_player)
        # Compute the adjustment factor for the prediction interval
        bound: cython.float = c * (
            sqrt(log(cython.cast(cython.float, max(1, N))) / cython.cast(cython.float, self.n_visits))
        )
        return value, bound

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-777777777, check=False)
    def get_value_imm(self, player: cython.short, imm_alpha: cython.float, max_player: cython.short) -> cython.float:
        simulation_mean: cython.float = self.v[player - 1] / self.n_visits

        if imm_alpha > 0.0:
            # Max player is the player that is maximizing overall, not just in this node
            if player == max_player:
                return ((1.0 - imm_alpha) * simulation_mean) + (imm_alpha * self.im_value)
            else:  # Subtract the im value
                return ((1.0 - imm_alpha) * simulation_mean) - (imm_alpha * self.im_value)
        else:
            return simulation_mean

    def __str__(self):
        root_mark = (
            f"(Root) AD:{self.anti_decisive:<1}" if self.action == () else f"{Fore.BLUE}A: {str(self.action):<10}"
        )

        solved_bg = ""
        if self.solved_player == 1:
            solved_bg = Back.LIGHTCYAN_EX
        elif self.solved_player == 2:
            solved_bg = Back.LIGHTWHITE_EX
        elif self.draw:
            solved_bg = Back.LIGHTGREEN_EX

        im_str = f"{Fore.WHITE}IM:{self.im_value:6.3f} " if abs(self.im_value) != INFINITY else ""

        # This is flipped because I want to see it in view of the parent
        value = self.v[(3 - self.player) - 1] / max(1, self.n_visits)

        return (
            f"{solved_bg}"
            f"{root_mark} "
            f"{Fore.GREEN}P: {self.player:<1} " + im_str + f"{Fore.YELLOW}EV: {self.eval_value:5.2f} "
            f"{Fore.CYAN}EXP: {'True' if self.expanded else 'False':<3} "
            f"{Fore.MAGENTA}SOLVP: {self.solved_player:<3} "
            f"{Fore.MAGENTA}DRW: {self.draw:<3} "
            f"{Fore.WHITE}VAL: {value:2.3f} "
            f"{Back.YELLOW + Fore.BLACK}NVIS: {self.n_visits:,}{Back.RESET + Fore.RESET}"
        )


@cython.cclass
class MCTSPlayer:
    player: cython.short
    num_simulations: cython.int
    e_g_subset: cython.int
    early_term: cython.bint
    dyn_early_term: cython.bint
    early_term_turns: cython.int
    early_term_cutoff: cython.float
    e_greedy: cython.bint
    prog_bias: cython.bint
    pb_weight: cython.float
    imm: cython.bint
    imm_alpha: cython.float
    debug: cython.bint
    roulette: cython.bint
    roulette_eps: cython.float
    dyn_early_term_cutoff: cython.float
    c: cython.float
    epsilon: cython.float
    max_time: cython.float
    eval_params: cython.double[:]
    root: Node
    reuse_tree: cython.bint
    random_top: cython.int
    name: cython.str

    # The highest evaluation value seen throughout the game (for normalisation purposes later on)
    max_eval: cython.float
    # The average depth of the playouts, for science
    avg_po_moves: cython.float
    playout_terminals: cython.int
    playout_draws: cython.int
    # The average number of playouts per second
    avg_pos_ps: cython.float
    n_moves: cython.int
    max_depth: cython.int
    avg_depth: cython.int
    # Research additions
    ab_p1: cython.short
    ab_p2: cython.short
    k_factor: cython.float

    def __init__(
        self,
        player: int,
        eval_params: cython.double[:],
        transposition_table_size: int = 2**16,
        num_simulations: int = 0,
        max_time: int = 0,
        c: float = 1.0,
        dyn_early_term_cutoff: float = 0.0,
        early_term_turns: int = 0,
        early_term_cutoff: float = 0.0,
        epsilon: float = 0.0,
        e_g_subset: int = 20,
        roulette_epsilon: float = 0.0,
        imm_alpha: float = 0.0,
        ab_p1: int = 0,
        ab_p2: int = 0,
        k_factor: float = 1.0,
        pb_weight: float = 0.0,
        reuse_tree: bool = True,
        debug: bool = False,
        random_top: int = 0,
        name: str = "",
    ):
        self.player = player
        self.dyn_early_term = dyn_early_term_cutoff > 0.0
        self.dyn_early_term_cutoff = dyn_early_term_cutoff
        self.early_term_cutoff = early_term_cutoff
        self.e_greedy = epsilon > 0.0
        self.epsilon = epsilon
        self.roulette = roulette_epsilon > 0.0
        self.roulette_eps = roulette_epsilon
        self.e_g_subset = e_g_subset
        self.early_term = early_term_cutoff > 0.0 or early_term_turns > 0
        self.early_term_turns = early_term_turns
        self.c = c
        self.eval_params = eval_params

        # Let the enable/disable of
        self.imm = imm_alpha > 0.0
        self.imm_alpha = imm_alpha
        self.prog_bias = pb_weight > 0.0
        self.pb_weight = pb_weight

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

        # Variables for debugging and science
        self.avg_po_moves = 0
        self.max_eval = -99999999.9
        self.avg_pos_ps = 0
        self.n_moves = 0
        self.avg_depth = 0
        self.max_depth = 0

        self.ab_p1 = ab_p1
        self.ab_p2 = ab_p2
        self.k_factor = k_factor

        self.reuse_tree = reuse_tree
        self.random_top = random_top
        self.name = name

    @cython.ccall
    def best_action(self, state: GameState) -> cython.tuple:
        assert state.player == self.player, "The player to move does not match my max player"

        self.n_moves += 1

        if self.reuse_tree:
            # Check if we can reutilize the root
            # If the root is None then this is either the first move, or something else..
            if state.REUSE and self.root is not None and self.root.expanded:
                child: Node
                if DEBUG:
                    print(f"Checking children of root node {str(self.root)} with {len(self.root.children)} children")
                    print(f"Last action: {str(state.last_action)}")

                children: cython.list = self.root.children
                self.root = None  # In case we cannot find the action, mark the root as None to assert

                for child in children:
                    if child.action == state.last_action:
                        self.root = child

                        if DEBUG:
                            print(f"Reusing root node {str(child)}")

                        self.root.action = ()  # This is what identifies the root node
                        break
            elif not state.REUSE or (self.root is not None and not self.root.expanded):
                # This sets the same condition as the one in the if statement above, make sure that a new root is generated
                self.root = None
        else:
            self.root = None

        if self.root is None:
            # Reset root for new round of MCTS
            self.root: Node = Node(state.player, (), self.player)

        if not self.root.expanded:
            self.root.add_all_children(
                init_state=state,
                max_player=self.player,
                prog_bias=self.prog_bias,
                imm=self.imm,
                eval_params=self.eval_params,
            )

        start_time: cython.long = curr_time()
        i: cython.int = 0

        if self.num_simulations:
            for i in range(self.num_simulations):
                self.simulate(state)
                # The root is solved, no need to look any further, since we have a winning/losing move
                if self.root.solved_player != 0:
                    break
                if self.root.anti_decisive:
                    break
                if self.root.draw:
                    break
        else:
            counter: cython.int = 0
            while True:
                self.simulate(state)
                i += 1
                # The root is solved, no need to look any further, since we have a winning/losing move
                if self.root.solved_player != 0:
                    break
                if self.root.anti_decisive:
                    break
                if self.root.draw:
                    break
                # Only check the time every once in a while to save cost
                counter += 1
                if counter > 500:
                    if curr_time() - start_time >= self.max_time:
                        break
                    counter = 0

        total_time: cython.long = curr_time() - start_time
        # print(f"{self.avg_po_moves=}")
        self.avg_po_moves = self.avg_po_moves / float(i + 1)
        self.avg_pos_ps += i / float(max(1, total_time))
        self.avg_depth = cython.cast(cython.int, self.avg_depth / (i + 1))

        if DEBUG:
            print(
                f"\n\n** ran {i+1:,} simulations in {format_time(total_time)}, {i / float(max(1, total_time)):,.0f} simulations per second **"
            )
            if self.root.solved_player != 0:
                print(f"*** Root node is solved for player {self.root.solved_player} ***")
            if self.root.anti_decisive:
                print("*** Anti-decisive move found ***")
            if self.root.draw:
                print("*** Proven draw ***")

        max_node: Node = None
        max_value: cython.float = -INFINITY
        n_children: cython.Py_ssize_t = len(self.root.children)
        all_loss: cython.bint = 1
        if self.random_top == 0:
            c_i: cython.short
            for c_i in range(0, n_children):
                node: Node = self.root.children[c_i]
                if node.solved_player == self.player:  # We found a winning move, let's go champ
                    max_node = node
                    max_value = node.n_visits
                    break

                if node.solved_player == 3 - self.player:  # We found a losing move
                    continue

                all_loss = 0
                value: cython.float = node.n_visits

                if value >= max_value:
                    max_node = node
                    max_value = value
        else:
            # Compute the number of nodes to select based on the percentage (random_top)
            n_top: cython.int = int(len(self.root.children) * (float(self.random_top) / 100.0))
            # This is used to kick-start experiments and ensure difference between experiments
            # Select a move at random from the best self.random_top moves
            top_nodes = sorted(self.root.children, key=get_node_visits, reverse=True)[:n_top]
            max_node = random.choice(top_nodes)

            if DEBUG:
                print(f"Randomly selected node from top {n_top} node: {str(max_node)}")
                for n in top_nodes:
                    print(f"{str(n)}")

        # In case there's no move that does not lead to a loss, we should return a random move
        if max_node is None:
            if DEBUG:
                if self.root.solved_player != (3 - self.player) and not self.root.draw:
                    print("Root not solved (no draw) for opponent and no max_node found!!")
                    print(f"Root node: {str(self.root)}")
                    print("\n".join([str(child) for child in self.root.children]))

            assert (
                all_loss or self.root.solved_player == (3 - self.player) or self.root.draw
            ), f"No max node found for root node {str(self.root)}\n{str(state)}"

            # return a random action if all children are losing moves
            if len(self.root.children) == 0:
                self.root = None
                return None, 0.0

            max_node = random.choice(self.root.children)

        if DEBUG:
            print("--*--" * 20)
            print(f"BEST NODE: {max_node}")
            print("-*=*-" * 15)
            next_state: GameState = state.apply_action(max_node.action)
            evaluation: cython.float = next_state.evaluate(params=self.eval_params, player=self.player, norm=False)
            norm_eval: cython.float = next_state.evaluate(params=self.eval_params, player=self.player, norm=True)
            self.max_eval = max(evaluation, self.max_eval)

            print(
                f"evaluation: {evaluation:.2f} / (normalized): {norm_eval:.4f} | max_eval: {self.max_eval:.1f} | Playout draws: {self.playout_draws:,} | Terminal Playout: {self.playout_terminals:,}"
            )
            print(
                f"max depth: {self.max_depth} | avg depth: {self.avg_depth:.2f}| avg. playout moves: {self.avg_po_moves:.2f} | avg. playouts p/s.: {self.avg_pos_ps / self.n_moves:,.0f} | {self.n_moves} moves played"
            )
            self.avg_po_moves = 0
            global ab_bound, ucb_bound
            print(
                f"alpha/beta bound used: {ab_bound:,} |  ucb bound used: {ucb_bound:,} | percentage: {((ab_bound) / max(ab_bound + ucb_bound, 1)) * 100:.2f}%"
            )
            ucb_bound = ab_bound = 0
            self.playout_draws = self.max_depth = self.avg_depth = 0
            self.playout_terminals = 0
            print("--*--" * 20)
            print(f":: {self.root} :: ")
            print(":: Children ::")
            comparator = ChildComparator()
            sorted_children = sorted(self.root.children, key=comparator, reverse=True)[:30]
            print("\n".join([str(child) for child in sorted_children]))
            print("--*--" * 20)
            if __debug__:
                if self.ab_p1 != 0:
                    plot_width = 120
                    plot_height = 26

                    plot_selected_bins(dynamic_bins, plot_width, plot_height)
                    # Clear bins
                    print("clearing bins, this takes a while..")
                    for _, bin_value in dynamic_bins.items():
                        del bin_value["bin"]
                        print(f"\rclearing {bin_value['label']}", end=" ")
                        bin_value["bin"] = DynamicBin(n_bins)

                    print("\rCollecting garbage")
                    gc.collect()
                    print("Done\n\n")

        max_action: cython.tuple = max_node.action

        if state.REUSE and self.reuse_tree:
            # Clean up memory by deleting all nodes except the best one
            for child in self.root.children:
                if child != max_node:
                    del child

            # For tree reuse, make sure that we can access the next action from the root
            self.root = max_node
        else:
            self.root = None

        return max_action, max_value  # return the most visited state

    @cython.cfunc
    @cython.nonecheck(False)
    def simulate(self, init_state: GameState):
        # Keep track of selected nodes
        selected: cython.list[Node] = [self.root]
        # Keep track of the state
        next_state: GameState = init_state
        is_terminal: cython.bint = init_state.is_terminal()

        expanded: cython.bint = 0  # Ensure expansion only occurs once

        # Start at the root
        node: Node = self.root
        prev_node: Node = None

        child: Node
        i: cython.int

        alpha: cython.float = -INFINITY
        beta: cython.float = INFINITY
        alpha_bounds: cython.float = -INFINITY
        beta_bounds: cython.float = INFINITY

        while not is_terminal and node.solved_player == 0:
            if node.expanded:
                if self.ab_p1 != 0:
                    # Check for new a/b bounds
                    if self.ab_p1 == 2:
                        if node.n_visits > 0 and prev_node is not None:
                            val, bound = node.get_value_with_uct_interval(
                                c=self.c,
                                player=self.player,
                                max_player=self.player,
                                imm_alpha=self.imm_alpha,
                                N=prev_node.n_visits,
                            )
                            if prev_node.player == self.player:
                                old_alpha: cython.float = alpha  # Store the old value of alpha
                                alpha = max(alpha, val - bound)
                                # Check if alpha was actually updated by comparing old and new values
                                if alpha > old_alpha:
                                    alpha_bounds = -bound
                            else:
                                old_beta: cython.float = beta  # Store the old value of beta
                                beta = min(beta, val + bound)
                                # Check if beta was actually updated by comparing old and new values
                                if beta < old_beta:
                                    beta_bounds = bound

                    if self.ab_p1 == 3:
                        # This is the version with the harder cutoffs
                        if node.n_visits > 0 and prev_node is not None:

                            val, bound = node.get_value_with_uct_interval(
                                c=self.c,
                                player=self.player,
                                max_player=self.player,
                                imm_alpha=self.imm_alpha,
                                N=prev_node.n_visits,
                            )

                            if val + bound <= beta and val - bound >= alpha:
                                # The node is within the bounds, check for new a/b bounds to use
                                if prev_node.player == self.player:
                                    old_alpha: cython.float = alpha  # Store the old value of alpha
                                    alpha = max(alpha, val - bound)
                                    # Check if alpha was actually updated by comparing old and new values
                                    if alpha > old_alpha:
                                        alpha_bounds = -bound
                                else:
                                    old_beta: cython.float = beta  # Store the old value of beta
                                    beta = min(beta, val + bound)
                                    # Check if beta was actually updated by comparing old and new values
                                    if beta < old_beta:
                                        beta_bounds = bound

                    prev_node = node
                    if __debug__:
                        # Keep track of the data used for each UCT call
                        if alpha != -INFINITY and beta != INFINITY:
                            dynamic_bins["alpha"]["bin"].add_data(alpha)
                            dynamic_bins["beta"]["bin"].add_data(beta)
                            dynamic_bins["alpha_bounds"]["bin"].add_data(alpha_bounds)
                            dynamic_bins["beta_bounds"]["bin"].add_data(beta_bounds)
                            dynamic_bins["k"]["bin"].add_data(beta - alpha)
                        elif alpha != -INFINITY:
                            dynamic_bins["alpha"]["bin"].add_data(alpha)
                            dynamic_bins["alpha_bounds"]["bin"].add_data(alpha_bounds)
                        elif beta != INFINITY:
                            dynamic_bins["beta"]["bin"].add_data(beta)
                            dynamic_bins["beta_bounds"]["bin"].add_data(beta_bounds)

                    node = node.uct(
                        self.c,
                        self.player,
                        self.pb_weight,
                        self.imm_alpha,
                        ab_p1=self.ab_p1,
                        ab_p2=self.ab_p2,
                        alpha=alpha if node.player == self.player else 1 - beta,
                        beta=beta if node.player == self.player else 1 - alpha,
                        k_factor=self.k_factor,
                        alpha_bounds=alpha_bounds if node.player == self.player else -beta_bounds,
                        beta_bounds=beta_bounds if node.player == self.player else -alpha_bounds,
                    )
                else:
                    node = node.uct(self.c, self.player, self.pb_weight, self.imm_alpha)

                assert node is not None, f"Node is None after UCT {prev_node}"

            elif not node.expanded and not expanded:
                prev_node = node
                # * Expand should always returns a node, even after adding the last node
                node = node.expand(
                    init_state=next_state,
                    max_player=self.player,
                    eval_params=self.eval_params,
                    prog_bias=self.prog_bias,
                    imm=self.imm,
                )
                expanded = 1

                assert (
                    node is not None or prev_node.solved_player != 0
                ), f"non-terminal Node is None after expansion {prev_node}\n{next_state.visualize(True)}"

            elif expanded:
                break

            if node is not None:
                next_state = next_state.apply_action(node.action)
                selected.append(node)
            else:
                node = prev_node
                is_terminal = True
                assert node.solved_player != 0, f"Node is None and not solved {prev_node}"

            is_terminal = next_state.is_terminal()

        # * Playout / Terminal node reached
        result: tuple[cython.float, cython.float]
        # If the node is neither terminal nor solved, then we need a playout
        if not is_terminal and node.solved_player == 0:
            # Do a random playout and collect the result
            result = self.play_out(next_state)
            if __debug__:
                if result == (0.5, 0.5):
                    self.playout_draws += 1
        else:
            if node.solved_player == 0 and is_terminal:
                # A terminal node is reached, so we can backpropagate the result of the state as if it was a playout
                result = next_state.get_result_tuple()
            elif node.solved_player == 1:
                result = (1.3, 0.0)
            elif node.solved_player == 2:
                result = (0.0, 1.3)

        # Keep track of the max depth of the tree
        self.max_depth = max(self.max_depth, len(selected))
        self.avg_depth += len(selected)

        # * Backpropagation
        for i in range(len(selected), 0, -1):  # Move backwards through the list
            node = selected[i - 1]

            # * Backpropagate the result of the playout
            if self.imm and node.expanded:
                if node.player == self.player:
                    node.im_value = -INFINITY  # Initialize to negative infinity
                    for i in range(len(node.children)):
                        child = node.children[i]
                        temp_im_value: cython.float = child.im_value
                        node.im_value = max(node.im_value, temp_im_value)
                else:
                    node.im_value = INFINITY  # Initialize to positive infinity
                    for i in range(len(node.children)):
                        child = node.children[i]
                        temp_im_value: cython.float = child.im_value
                        node.im_value = min(node.im_value, temp_im_value)

            node.v[0] += result[0]
            node.v[1] += result[1]

            node.n_visits += 1

    @cython.cfunc
    def play_out(self, state: GameState) -> cython.tuple[cython.float, cython.float]:
        turns: cython.short = 0
        while not state.is_terminal():
            # Early termination condition with a fixed number of turns
            if self.early_term and turns >= self.early_term_turns:
                # ! This assumes symmetric evaluation functions centered around 0!
                # * Figure out the a (max range) for each evaluation function
                evaluation: cython.float = state.evaluate(params=self.eval_params, player=1, norm=True)

                if evaluation >= self.early_term_cutoff:
                    return (1.0, 0.0)
                elif evaluation < -self.early_term_cutoff:
                    return (0.0, 1.0)
                else:
                    return (0.5, 0.5)

            # Dynamic Early termination condition, check every few turns if the evaluation has a certain value
            elif self.early_term == 0 and self.dyn_early_term == 1 and turns % 6 == 0:
                # ! This assumes symmetric evaluation functions centered around 0!
                # * Figure out the a (max range) for each evaluation function
                evaluation = state.evaluate(params=self.eval_params, player=1, norm=True)

                if evaluation >= self.dyn_early_term_cutoff:
                    return (1.0, 0.0)
                elif evaluation < -self.dyn_early_term_cutoff:
                    return (0.0, 1.0)

            best_action: cython.tuple = ()
            # With probability epsilon choose the best action from a subset of moves
            if self.e_greedy == 1 and uniform(0, 1) < self.epsilon:
                actions = state.get_legal_actions()
                actions = actions[: self.e_g_subset]

                max_value = -99999.99
                for i in range(len(actions)):
                    # Evaluate the new state in view of the player to move
                    value = state.apply_action(actions[i]).evaluate(
                        params=self.eval_params,
                        player=state.player,
                        norm=False,
                    )
                    # Keep track of the best action
                    if value > max_value:
                        max_value = value
                        best_action = actions[i]

            # With probability epsilon choose a move using roulette wheel selection based on the move ordering
            elif self.e_greedy == 0 and self.roulette == 1 and uniform(0, 1) <= self.roulette_eps:
                actions = state.get_legal_actions()
                best_action = random.choices(actions, weights=state.move_weights(actions), k=1)[0]

            # With probability 1-epsilon chose a move at random
            if best_action == ():
                best_action = state.get_random_action()

            state.apply_action_playout(best_action)

            self.avg_po_moves += 1
            turns += 1

        self.playout_terminals += 1
        # print("\n\n")
        # print(turns)
        # print(f"{state.visualize(True)}")
        # input("Press Enter to continue...")
        # Map the result to the players
        res_tuple: cython.tuple[int, int] = state.get_result_tuple()
        # multiply all 1's by 1.3 to reward true wins
        if res_tuple[0] == 1:
            return (1.3, 0.0)
        elif res_tuple[1] == 1:
            return (0.0, 1.3)

    def print_cumulative_statistics(self) -> str:
        return ""

    def __repr__(self):
        return (
            f"MCTS(name={self.name}, p={self.player}, eval_params={self.eval_params}, c={self.c}, "
            + f"d_early_co={self.dyn_early_term_cutoff}, early_t_t={self.early_term_turns}, eps={self.epsilon}, "
            + f"e_g_s={self.e_g_subset}, p_b_w={self.pb_weight}, imm_a={self.imm_alpha}, "
            + f"ab_p1/2={self.ab_p1}/{self.ab_p2})"
        )


class ChildComparator:
    def __call__(self, child):
        return child.n_visits


# Define a key function for sorting
def get_node_visits(node):
    return node.n_visits


@cython.cfunc
@cython.returns(cython.float)
@cython.exceptval(-88888888, check=False)  # Don't use 99999 because it's win/los
@cython.locals(
    is_max_player=cython.bint,
    v=cython.float,
    new_v=cython.float,
    actions=cython.list,
    move=cython.tuple,
    m=cython.short,
    i=cython.short,
    n_actions=cython.short,
)
def alpha_beta(
    state: GameState,
    alpha: cython.float,
    beta: cython.float,
    depth: cython.short,
    max_player: cython.short,
    eval_params: cython.double[:],
):
    if state.is_terminal():
        return state.get_reward(max_player)

    if depth == 0:
        return state.evaluate(params=eval_params, player=max_player, norm=True) + uniform(-0.0001, 0.0001)

    is_max_player = state.player == max_player
    # Move ordering
    actions = state.get_legal_actions()
    # actions = state.evaluate_moves(state.get_legal_actions())
    # actions = [(actions[i][0], actions[i][1]) for i in range(n_actions)] # ? I don't think that this line is needed here
    # actions.sort(key=itemgetter(1), reverse=is_max_player)
    # actions = [actions[i][0] for i in range(n_actions)] # ? Nor is this one needed here as long as we use actions[m][0] in the loop (instead of actions[m])

    v = -INFINITY if is_max_player else INFINITY
    for m in range(len(actions)):
        move = actions[m]

        new_v = alpha_beta(
            state=state.apply_action(move),
            alpha=alpha,
            beta=beta,
            depth=depth - 1,
            max_player=max_player,
            eval_params=eval_params,
        )
        # Update v, alpha or beta based on the player
        if (is_max_player and new_v > v) or (not is_max_player and new_v < v):
            v = new_v
        # Alpha-beta pruning
        if is_max_player:
            alpha = max(alpha, new_v)
        else:
            beta = min(beta, new_v)
        # Prune the branch
        if beta <= alpha:
            break

    return v


@cython.cfunc
@cython.returns(cython.float)
@cython.locals(
    actions=cython.list,
    move=cython.tuple,
    score=cython.float,
    stand_pat=cython.float,
    m=cython.short,
)
@cython.exceptval(-88888888, check=False)
def quiescence(
    state: GameState,
    stand_pat: cython.float,
    max_player: cython.short,
    eval_params: cython.double[:],
):
    actions = state.get_legal_actions()
    for m in range(len(actions)):
        move = actions[m]
        if state.is_capture(move):
            new_state = state.apply_action(move)
            new_stand_pat = new_state.evaluate(params=eval_params, player=max_player, norm=True) + uniform(
                -0.0001, 0.0001
            )
            score = -quiescence(new_state, new_stand_pat, max_player, eval_params)
            stand_pat = max(stand_pat, score)  # Update best_score if a better score is found

    return stand_pat  # Return the best score found


def plot_selected_bins(bins_dict, plot_width=140, plot_height=32):
    while True:
        print("Available statistics to plot:")
        for idx, key in enumerate(bins_dict.keys()):
            print(f"{idx + 1}. {bins_dict[key]['label']}")

        print("0. Continue")

        user_input = input("Enter the number fo the statistics to plot, or 0 to continue, j to just play, q to quit: ")

        if user_input == "0":
            break
        elif user_input == "q":
            quit()
        elif user_input.isdigit():
            selected_idx = int(user_input) - 1

            # Check if the selected index is within the range of available keys
            if 0 <= selected_idx < len(bins_dict):
                selected_key = list(bins_dict.keys())[selected_idx]
                selected_bin_value = bins_dict[selected_key]

                selected_bin_value["bin"].plot_bin_counts(selected_bin_value["label"])
                selected_bin_value["bin"].plot_time_series(
                    selected_bin_value["label"], plot_width, plot_height, median=False
                )
            else:
                print("Invalid input. The number is out of range. Please try again.")
        else:
            print("Invalid input. Please enter a number or 'q' to quit.")


# if DEBUG:
#     bins_dict["p1_value_bins"]["bin"].add_data(val if node.player == 1 else -val)
#     bins_dict["p2_value_bins"]["bin"].add_data(val if node.player == 2 else -val)
#     bins_dict["bound_bins"]["bin"].add_data(bound)

#     if alpha_val[player_i] != -INFINITY:
#         bins_dict["alpha_bins"]["bin"].add_data(alpha_val[player_i])
#     if beta_val[player_i] != INFINITY:
#         bins_dict["beta_bins"]["bin"].add_data(beta_val[player_i])

#     bins_dict["ab_visits_bins"]["bin"].add_data(node.n_visits)
#     bins_dict["ab_depth_bins"]["bin"].add_data(depth)

#     if alpha_val[player_i] != -INFINITY and beta_val[player_i] != INFINITY:
#         bins_dict["dist_bins"]["bin"].add_data((beta[player_i] - alpha[player_i]))
#     #  ======
#     if alpha_val[0] != -INFINITY:
#         bins_dict["alpha_1_bins"]["bin"].add_data(alpha[0])
#         bins_dict["beta_1_bins"]["bin"].add_data(beta[0])
#         if beta[0] != INFINITY:
#             bins_dict["1_dist_bins"]["bin"].add_data(beta[0] - alpha[0])
#     if alpha_val[1] != -INFINITY:
#         bins_dict["alpha_2_bins"]["bin"].add_data(alpha[1])
#         bins_dict["beta_2_bins"]["bin"].add_data(beta[1])
#         if beta[1] != INFINITY:
#             bins_dict["2_dist_bins"]["bin"].add_data(beta[1] - alpha[1])

# bins_dict = {
#     "p1_value_bins": {"bin": DynamicBin(n_bins), "label": "p1 ab Values"},
#     "p2_value_bins": {"bin": DynamicBin(n_bins), "label": "p2 ab Values"},
#     "bound_bins": {"bin": DynamicBin(n_bins), "label": "Bounds around a/b"},
#     "alpha_bins": {"bin": DynamicBin(n_bins), "label": "Effective Alpha values"},
#     "beta_bins": {"bin": DynamicBin(n_bins), "label": "Effective Beta values"},
#     "ab_visits_bins": {"bin": DynamicBin(n_bins), "label": "a/b/ Node visit counts"},
#     "ab_depth_bins": {"bin": DynamicBin(n_bins), "label": "Depth of a/b nodes"},
#     "dist_bins": {"bin": DynamicBin(n_bins), "label": "Difference between Beta and Alpha"},
#     "alpha_1_bins": {"bin": DynamicBin(n_bins), "label": "Alpha values for Player 1 nodes"},
#     "beta_1_bins": {"bin": DynamicBin(n_bins), "label": "Beta values for Player 1 nodes"},
#     "1_dist_bins": {"bin": DynamicBin(n_bins), "label": "Difference between Beta and Alpha for Player 1"},
#     "alpha_2_bins": {"bin": DynamicBin(n_bins), "label": "Alpha values for Player 2 nodes"},
#     "beta_2_bins": {"bin": DynamicBin(n_bins), "label": "Beta values for Player 2 nodes"},
#     "2_dist_bins": {"bin": DynamicBin(n_bins), "label": "Difference between Beta and Alpha for Player 2"},
#     "ab_uct_bins": {"bin": DynamicBin(n_bins), "label": "Best UCB values"},
# }
