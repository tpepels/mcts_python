# cython: language_level=3

from cmath import isnan
import gc
import math
import random
from random import random as rand_float
from colorama import Back, Fore, init
from includes.dynamic_bin import DynamicBin

init(autoreset=True)

import cython
from cython.cimports.libc.time import time
from cython.cimports.libc.math import sqrt, log, INFINITY, isfinite
from cython.cimports.includes import GameState, win, loss, hash_tuple

from util import format_time

just_play = cython.declare(cython.bint, False)
prunes = cython.declare(cython.int, 0)
non_prunes = cython.declare(cython.int, 0)
ab_bound = cython.declare(cython.int, 0)
ucb_bound = cython.declare(cython.int, 0)

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
    r_float: cython.float = rand_float()
    return a + (b - a) * r_float


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
    action = cython.declare(cython.tuple, visibility="public")
    n_children = cython.declare(cython.short, visibility="public")
    amaf_visits = cython.declare(cython.int, visibility="public")
    amaf_wins = cython.declare(cython.float, visibility="public")
    # Private fields
    draw: cython.bint
    eval_value: cython.float
    actions: cython.list[cython.tuple]  # A list of actions that is used to expand the node

    def __cinit__(self, player: cython.short, action: cython.tuple, max_player: cython.short):
        # The action that led to this state, needed for root selection
        self.action = action
        self.player = player
        # Now we know whether we are minimizing or maximizing
        self.im_value = loss - 1 if self.player == max_player else win + 1
        self.eval_value = 0.0
        self.expanded = 0
        self.solved_player = 0
        self.v = [0.0, 0.0]  # Values for player-1
        self.n_visits = 0
        self.n_children = 0  # An unexpanded node has no children
        self.actions = []
        self.children = []

    @cython.cfunc
    def uct(
        self,
        c: cython.float,
        max_player: cython.short,
        pb_weight: cython.float = 0.0,
        imm_alpha: cython.float = 0.0,
        rave_k: cython.float = 0.0,
        ab_p1: cython.short = 0,
        ab_p2: cython.short = 0,
        alpha: cython.float = -INFINITY,
        beta: cython.float = INFINITY,
        k_factor: cython.float = 1.0,
        alpha_bounds: cython.float = 0.0,
        beta_bounds: cython.float = 0.0,
    ) -> Node:
        assert self.expanded, "Trying to uct a node that is not expanded"
        assert self.n_children > 0, "Trying to uct a node with no children"
        global ucb_bound, ab_bound

        p_n: cython.float = cython.cast(cython.float, max(1, self.n_visits))
        log_p_n: cython.float = log(p_n)

        if ab_p1 != 0 and isfinite(alpha) and isfinite(beta):
            k: cython.float = ((beta + beta_bounds) - (alpha - alpha_bounds)) * (1 - (beta_bounds - alpha_bounds))
            # TODO Deze versie moet je nog testen
            # k: cython.float = (beta + beta_bounds) - (alpha - alpha_bounds)
            if k != 0:
                ab_bound += 1
                # if ab_p1 == 1:
                #     log_p_n = log((1 - k) * p_n)  # TODO Deze versie moet je nog testen
                # if ab_p1 == 2:
                log_p_n *= (k_factor**2) * log((1 - k) * p_n)
                # elif ab_p1 == 3:
                # log_p_n *= (c**2) * log((1 - k) * p_n)
                # if ab_p1 == 4:
                # log_p_n *= (k_factor**2) * log((1 + k) * p_n)
                # elif ab_p1 == 5:
                # log_p_n *= (c**2) * log((1 + k) * p_n)
            else:
                ucb_bound += 1
        else:
            ucb_bound += 1

        if rave_k > 0.0:
            # Calculate Beta for the RAVE balance (assumes c_n and total_visits are defined)
            beta: cython.float = sqrt(rave_k / (3 * p_n + rave_k))

        # Move through the children to find the one with the highest UCT value
        if imm_alpha > 0.0:
            # Find the max and min values for the children to normalize them
            max_imm: cython.float = loss - 1
            min_imm: cython.float = win + 1
            for ci in range(self.n_children):
                child: Node = self.children[ci]
                if child.im_value > max_imm:
                    max_imm = child.im_value
                if child.im_value < min_imm:
                    min_imm = child.im_value

        selected_index: cython.short = -1
        best_val: cython.float = -INFINITY
        children_lost: cython.short = 0
        children_draw: cython.short = 0

        ci: cython.short
        for ci in range(self.n_children):
            child: Node = self.children[ci]
            assert child.action is not None, f"Child action is None! {str(self)}"

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

            if rave_k > 0.0 and child.amaf_visits > 1 and beta > 0.0:
                # Calculate the UCT value with RAVE
                # ! AMAF wins are in view of the parent player, so no need to switch
                uct_val = (
                    ((1 - beta) * (child.get_value_imm(self.player, imm_alpha, max_player, min_imm, max_imm)))
                    + (beta * (child.amaf_wins / cython.cast(cython.float, child.amaf_visits)))
                    + (c * sqrt(log_p_n / c_n))
                )
            else:
                # ! The log of p_n is only calculated once, so we can use it here as is.
                uct_val: cython.double = child.get_value_imm(self.player, imm_alpha, max_player, min_imm, max_imm) + (
                    c * sqrt(log_p_n / c_n)
                )

            if pb_weight > 0.0:
                uct_val += pb_weight * (cython.cast(cython.float, child.eval_value) / (1.0 + c_n))

            assert not isnan(
                uct_val
            ), f"UCT value is NaN!.\nNode: {str(self)}\nChild: {str(child)}\n{(self.player, imm_alpha, max_player, min_imm, max_imm)}\n{c * sqrt(log_p_n / c_n)}\n{str(child.children)}"

            rand_fact: cython.float = uniform(-0.0001, 0.0001)

            # Find the highest UCT value
            if (uct_val + rand_fact) >= best_val:
                selected_index = ci
                best_val = uct_val

        # It may happen that somewhere else in the tree all my children have been found to be proven losses.
        if children_lost == self.n_children:
            # Since all children are losses, we can mark this node as solved for the opponent
            # We do this here because the node may have been expanded and solved elsewhere in the tree
            self.solved_player = 3 - self.player
            # just return a random child, they all lead to a loss anyway
            return random.choice(self.children)

        elif children_lost == (self.n_children - 1):
            self.check_loss_node()

            if self.solved_player == 0 or self.solved_player == self.player:
                # There's only one move that does not lead to a loss. This is an anti-decisive move.
                self.anti_decisive = 1

        # Proven draw
        elif children_draw == self.n_children:
            self.draw = 1

        assert selected_index is not None, f"No child selected in UCT! {str(self)}"
        assert (
            0 <= selected_index < self.n_children
        ), f"Selected index out of bounds {selected_index} {self.n_children}"

        return self.children[selected_index]

    @cython.cfunc
    def expand(
        self,
        init_state: GameState,
        max_player: cython.short,
        eval_params: cython.float[:],
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
            self.n_children = len(self.children)

        while len(self.actions) > 0:
            # Get a random action from the list of previously generated actions
            action: cython.tuple = self.actions.pop()

            new_state: GameState = init_state.apply_action(action)
            child: Node = Node(new_state.player, action, max_player)

            # This works because we pop the actions
            self.children[(self.n_children - len(self.actions)) - 1] = child

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
            else:
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
                    # At first, the im value of the child is the same as the evaluation value,
                    # when searching deeper, it becomes the min/max of the subtree
                    child.im_value = new_state.evaluate(
                        player=max_player,
                        norm=True,
                        params=eval_params,
                    ) + uniform(-0.00001, 0.00001)
                    child.n_visits += 1
                    # ! This make sure that we only overwrite if we are sure we've found a "better" move
                    if (self.player != max_player and child.im_value < self.im_value) or (
                        self.player == max_player and child.im_value > self.im_value
                    ):
                        # The child improves the current evaluation, hence we can update the evaluation value
                        self.im_value = child.im_value

            return child  # Return the chlid, this is the node we will explore next.

        # If we've reached this point, then the node is fully expanded so we can switch to UCT selection
        self.expanded = 1

        # We reached a node with no successors, this means loss/draw depending on the game
        if self.n_children == 0:
            if self.solved_player == 0:
                raise ValueError(f"Node {str(self)} has no children, but is not solved {init_state.visualize(True)}")

            return None

        # Check if all my nodes lead to a loss.
        self.check_loss_node()

        if imm:
            # Do the full minimax back-up
            best_node: Node
            best_im: cython.float = loss - 1 if self.player == max_player else win + 1
            # print(best_im)
            i: cython.short
            child: Node
            for i in range(self.n_children):
                child = self.children[i]
                child_im: cython.float = child.im_value
                # * Minimize or Maximize my im value over all children
                if (self.player == max_player and child_im > best_im) or (
                    self.player != max_player and child_im < best_im
                ):
                    best_im = child_im
                    best_node = child
                    # print(f"New best im value: {best_im} from {str(best_node)}")
                # else:
                #     print(f"Child {str(child)} is not the best im value")

            # Now overwrite the im value of the node with the best im value
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
        eval_params: cython.float[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
    ):
        """
        Adds a previously expanded node's children to the tree
        """
        while not self.expanded:
            self.expand(init_state, max_player=max_player, prog_bias=prog_bias, imm=imm, eval_params=eval_params)

    @cython.cfunc
    @cython.locals(i=cython.short)
    def check_loss_node(self) -> cython.void:
        # I'm solved, nothing to see here.
        if self.solved_player != 0:
            return
        opp: cython.short = 3 - self.player
        child: Node
        for i in range(self.n_children):
            child = self.children[i]
            # If all children lead to a loss, then we will not return from the function
            if child.solved_player != opp:
                return

        self.solved_player = opp

    @cython.cfunc
    @cython.inline
    @cython.exceptval(-777777777, check=False)
    def get_value_imm(
        self,
        player: cython.short,
        imm_alpha: cython.float,
        max_player: cython.short,
        min_imm: cython.float,
        max_imm: cython.float,
    ) -> cython.float:
        if imm_alpha > 0.0:
            # Since imm values are in view of the maximizing player, flip the sign for the minimizing player
            sign: cython.float = 1.0 if player == max_player else -1.0
            if max_imm > min_imm:
                # Return the value including the imm value, which is normalized between 0 and 1
                return ((1.0 - imm_alpha) * (self.v[player - 1] / self.n_visits)) + (
                    imm_alpha * (sign * ((self.im_value - min_imm) / (max_imm - min_imm)))
                )
            else:
                # If the max and min values are the same, then the imm value is the same for all children
                return ((1.0 - imm_alpha) * (self.v[player - 1] / self.n_visits)) + (imm_alpha * sign * self.im_value)
        else:

            return self.v[player - 1] / self.n_visits

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
            f"{Fore.GREEN}P: {self.player:<1} " + im_str + f"{Fore.YELLOW}EV: {self.eval_value:2.3f} "
            f"{Fore.BLUE}AMAF_v: {self.amaf_visits:<6} AMAV_w: {self.amaf_wins:5.0f} "
            f"{Fore.CYAN}EXP: {'True' if self.expanded else 'False':<4} "
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
    early_term_turns: cython.short
    early_term_cutoff: cython.float
    prog_bias: cython.bint
    pb_weight: cython.float
    imm: cython.bint
    imm_alpha: cython.float
    # Mast and e greedy both use epsilon
    mast: cython.bint
    mast_visits: cython.uint[10000][2]
    # mast_visits_2: cython.uint[10000]
    mast_values: cython.float[10000][2]
    # mast_values_2: cython.float[10000]
    e_greedy: cython.bint
    epsilon: cython.float

    dyn_early_term_cutoff: cython.float
    c: cython.float
    max_time: cython.float
    eval_params: cython.float[:]
    root: Node
    reuse_tree: cython.bint
    random_top: cython.int
    name: cython.str
    # 0 <- number of visits
    # 1 <- average score
    move_selection: cython.short

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
    mast_moves: cython.int
    rave_k: cython.float
    rave: cython.bint
    # Research additions
    ab_p1: cython.short
    ab_p2: cython.short
    k_factor: cython.float

    def __init__(
        self,
        player: int,
        eval_params: cython.float[:],
        transposition_table_size: int = 2**16,
        num_simulations: int = 0,
        max_time: int = 0,
        c: float = 1.0,
        dyn_early_term_cutoff: float = 0.0,
        early_term_turns: int = 0,
        early_term_cutoff: float = 0.0,
        epsilon: float = 0.0,
        e_greedy: bool = False,
        mast: bool = False,
        rave_k: float = 0,
        e_g_subset: int = 20,
        imm_alpha: float = 0.0,
        ab_p1: int = 0,
        ab_p2: int = 0,
        k_factor: float = 1.0,
        pb_weight: float = 0.0,
        reuse_tree: bool = True,
        debug: bool = False,  # * Because we use the __debug__ flag, this no longer has any function
        random_top: int = 0,
        move_selection: int = 0,
        name: str = "",
    ):
        self.player = player

        self.dyn_early_term = dyn_early_term_cutoff >= 0.001
        self.dyn_early_term_cutoff = dyn_early_term_cutoff

        self.e_greedy = e_greedy
        self.mast = mast
        self.epsilon = epsilon
        self.e_g_subset = e_g_subset

        self.early_term = early_term_cutoff >= 0.001 and early_term_turns > 0
        self.early_term_cutoff = early_term_cutoff
        self.early_term_turns = early_term_turns

        self.c = c
        self.eval_params = eval_params
        self.move_selection = move_selection

        # Let the enable/disable of
        self.imm = imm_alpha >= 0.001
        self.imm_alpha = imm_alpha
        self.prog_bias = pb_weight >= 0.001
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

        # Variables for debugging and science
        self.avg_po_moves = 0
        self.max_eval = -99999999.9
        self.avg_pos_ps = 0
        self.n_moves = 0
        self.avg_depth = 0
        self.max_depth = 0
        self.mast_moves = 0
        if rave_k >= 0.01:
            self.rave = True
        self.rave_k = rave_k

        self.ab_p1 = ab_p1
        self.ab_p2 = ab_p2
        self.k_factor = k_factor

        self.reuse_tree = reuse_tree
        self.random_top = random_top
        self.name = name

    @cython.ccall
    def best_action(self, state: GameState) -> cython.tuple:
        assert state.player == self.player, "The player to move does not match my max player"
        # Reset the MAST values
        if self.mast:
            self.mast_visits = [[0] * 10000 for _ in range(2)]
            self.mast_values = [[0.0] * 10000 for _ in range(2)]
            # Get the legal actions for the current state
            actions: cython.list[cython.tuple] = state.get_legal_actions()
            # Insert the move evaluations into the MAST values to kickstart it
            c_i: cython.short
            for c_i in range(len(actions)):
                action_hash: cython.uint = hash_tuple(actions[c_i], 10000)
                # Evaluate each action and use that as the initial value for MAST for my player
                new_state: GameState = state.apply_action(actions[c_i])
                value: cython.float = new_state.evaluate(player=self.player, norm=True, params=self.eval_params)
                self.mast_values[self.player - 1][action_hash] = value * 5
                self.mast_visits[self.player - 1][action_hash] = 5

        self.n_moves += 1

        if self.reuse_tree:
            # Check if we can reutilize the root
            # If the root is None then this is either the first move, or something else..
            if state.REUSE and self.root is not None and self.root.expanded:
                child: Node
                if __debug__:
                    print(f"Checking children of root node {str(self.root)} with {len(self.root.children)} children")
                    print(f"Last action: {str(state.last_action)}")

                children: cython.list = self.root.children
                self.root = None  # In case we cannot find the action, mark the root as None to assert

                for child in children:
                    if child.action == state.last_action:
                        self.root = child

                        if __debug__:
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
            # Make sure that the root is fully expanded
            self.root.add_all_children(
                init_state=state,
                max_player=self.player,
                prog_bias=self.prog_bias,
                imm=self.imm,
                eval_params=self.eval_params,
            )

        start_time: cython.double = time(cython.NULL)
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
                if counter >= 2000:
                    counter = 0
                    if time(cython.NULL) - start_time >= self.max_time:
                        break

        total_time: cython.long = cython.cast(cython.long, time(cython.NULL) - start_time)
        self.avg_po_moves = self.avg_po_moves / float(i + 1)
        self.avg_pos_ps += i / float(max(1, total_time))
        self.avg_depth = cython.cast(cython.int, self.avg_depth / (i + 1))

        n_children: cython.short = self.root.n_children

        all_loss: cython.bint = 1
        max_node: Node = None
        value: cython.float = 0
        max_value: cython.float = -INFINITY

        if self.random_top == 0:
            c_i: cython.short

            if self.imm:
                # Find the max and min values for the children to normalize them
                max_imm: cython.float = loss - 1
                min_imm: cython.float = win + 1
                for c_i in range(n_children):
                    child: Node = self.root.children[c_i]
                    if child.im_value > max_imm:
                        max_imm = child.im_value
                    if child.im_value < min_imm:
                        min_imm = child.im_value

            for c_i in range(0, n_children):

                node: Node = self.root.children[c_i]
                if node.solved_player == self.player:  # We found a winning move, let's go champ
                    max_node = node
                    max_value = node.n_visits
                    break

                if node.solved_player == 3 - self.player:  # We found a losing move
                    continue

                all_loss = 0
                if self.move_selection == 0:
                    value = node.n_visits
                elif self.move_selection == 1:
                    value = node.get_value_imm(self.player, self.imm_alpha, self.player, min_imm, max_imm)

                if value >= max_value:
                    max_node = node
                    max_value = value
        else:
            # * For initial "randomized" openings
            # Compute the number of nodes to select based on the percentage (random_top)
            n_top: cython.int = int(self.root.n_children * (float(self.random_top) / 100.0))
            # This is used to kick-start experiments and ensure difference between experiments
            # Select a move at random from the best self.random_top moves
            top_nodes: cython.list = sorted(self.root.children, key=get_node_visits, reverse=True)[:n_top]
            max_node: Node = random.choice(top_nodes)

            if __debug__:
                print(f"Randomly selected node from top {n_top} node: {str(max_node)}")
                for n in top_nodes:
                    print(f"{str(n)}")

        # In case there's no move that does not lead to a loss, we should return a random move
        if max_node is None:
            # This should not happen. Do not turn this into an assertion.
            if not (all_loss or self.root.solved_player == (3 - self.player) or self.root.draw):
                raise ValueError(f"No max node found for root node {str(self.root)}\n{str(state)}")

            # * We should only be here in case of a loss or a draw
            # return a random action if all children are losing moves
            if self.root.n_children == 0:
                self.root = None
                return None, 0.0

            max_node = random.choice(self.root.children)

        print(
            f"\nRan {i+1:,} simulations in {format_time(total_time)}, {i / float(max(1, total_time)):,.0f} simulations p.s.\n"
        )

        if __debug__:
            # Print debug information if the debug flag is set
            self.print_debug_info(i, total_time, max_node, state)

        max_action: cython.tuple = max_node.action

        if state.REUSE and self.reuse_tree:
            # * Keep track of the node that we played, next move, the root will be the opponent's move.
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
        p1_actions: cython.list[cython.tuple] = []
        p2_actions: cython.list[cython.tuple] = []
        # Keep track of the state
        next_state: GameState = init_state
        is_terminal: cython.bint = init_state.is_terminal()

        expanded: cython.bint = 0  # Ensure expansion only occurs once

        # Start at the root
        node: Node = self.root
        # Prev_node is the parent for alpha/beta selection
        prev_node: Node = None

        child: Node
        i: cython.int

        alpha: cython.float = -INFINITY
        beta: cython.float = INFINITY

        alpha_val: cython.float = -INFINITY
        beta_val: cython.float = INFINITY

        alpha_bounds: cython.float = -INFINITY
        beta_bounds: cython.float = INFINITY

        while not is_terminal and node.solved_player == 0:
            if node.expanded:
                if self.ab_p1 != 0:
                    # Check for new a/b bounds
                    if node.n_visits > 0 and prev_node is not None:

                        if self.imm:
                            # Find the max and min values for the children to normalize them
                            max_imm: cython.float = loss - 1
                            min_imm: cython.float = win + 1

                            for ci in range(prev_node.n_children):
                                child: Node = prev_node.children[ci]
                                if child.im_value > max_imm:
                                    max_imm = child.im_value
                                if child.im_value < min_imm:
                                    min_imm = child.im_value

                        val: cython.float = node.get_value_imm(
                            self.player, self.imm_alpha, self.player, min_imm, max_imm
                        )

                        # Compute the adjustment factor for the prediction interval
                        bound: cython.float = self.k_factor * (
                            sqrt(
                                log(cython.cast(cython.float, prev_node.n_visits))
                                / cython.cast(cython.float, node.n_visits)
                            )
                        )

                        if prev_node.player == self.player:
                            # Check if alpha was actually updated by comparing old and new values
                            if alpha < val - bound:
                                alpha_bounds = -bound
                                alpha_val = val
                                alpha = val - bound
                        else:
                            # Check if beta was actually updated by comparing old and new values
                            if beta > val + bound:
                                beta_bounds = bound
                                beta_val = val
                                beta = val + bound

                    prev_node = node

                    if __debug__ and not just_play:
                        # Keep track of the data used for each UCT call
                        if alpha != -INFINITY and beta != INFINITY:
                            dynamic_bins["alpha"]["bin"].add_data(alpha)
                            dynamic_bins["beta"]["bin"].add_data(beta)
                            dynamic_bins["alpha_bounds"]["bin"].add_data(alpha_bounds)
                            dynamic_bins["beta_bounds"]["bin"].add_data(beta_bounds)
                            # dynamic_bins["k"]["bin"].add_data(beta - alpha)
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
                        rave_k=self.rave_k if self.rave else 0.0,
                        ab_p1=self.ab_p1,
                        ab_p2=self.ab_p2,
                        alpha=alpha_val if node.player == self.player else 1 - beta_val,
                        beta=beta_val if node.player == self.player else 1 - alpha_val,
                        k_factor=self.k_factor,
                        alpha_bounds=alpha_bounds if node.player == self.player else -beta_bounds,
                        beta_bounds=beta_bounds if node.player == self.player else -alpha_bounds,
                    )
                else:
                    node = node.uct(
                        self.c, self.player, self.pb_weight, self.imm_alpha, rave_k=self.rave_k if self.rave else 0.0
                    )

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
                # For MAST, keep track of the actions taken
                if self.mast:
                    if next_state.player == 1 and node.action != None:  # ! The player that is GOING TO MAKE the move
                        p1_actions.append(node.action)
                    elif node.action != None:
                        p2_actions.append(node.action)

                next_state = next_state.apply_action(node.action)
                selected.append(node)
            else:
                node = prev_node
                is_terminal = True
                assert node.solved_player != 0, f"Node is None and not solved {prev_node}"

            is_terminal = next_state.is_terminal()

        # * Playout / Terminal node reached
        result: cython.tuple[cython.float, cython.float]
        # If the node is neither terminal nor solved, then we need a playout
        if not is_terminal and node.solved_player == 0:
            # Do a random playout and collect the result
            result = self.play_out(next_state, p1_actions, p2_actions)
        else:
            if node.solved_player == 0 and is_terminal:  # A draw
                # A terminal node is reached, so we can backpropagate the result of the state as if it was a playout
                result = next_state.get_result_tuple()
            elif node.solved_player == 1:
                result = (1.5, 0.0)
            elif node.solved_player == 2:
                result = (0.0, 1.5)
            else:
                raise ValueError(f"Node is not solved nor terminal {str(node)}")

        is_draw: cython.bint = math.isclose(result[0], 0.5)

        if __debug__ and is_draw:
            self.playout_draws += 1

        if self.mast and not is_draw:
            # Update the MAST values and visits for both players
            self.update_player_mast_values(p1_actions, 0, result[0])
            self.update_player_mast_values(p2_actions, 1, result[1])

        # Keep track of the max depth of the tree
        if __debug__:
            self.max_depth = max(self.max_depth, len(selected))
            self.avg_depth += len(selected)

        if self.ab_p2 == 5 and isfinite(alpha) and isfinite(beta):
            # This method scales playout rewards by the width of the alpha beta bounds
            k: cython.float = 1 - (beta - alpha)
            result[0] *= k
            result[1] *= k

        # * Backpropagation
        j: cython.int
        for i in range(len(selected), 0, -1):  # Move backwards through the list
            node = selected[i - 1]
            node.v[0] += result[0]
            node.v[1] += result[1]
            node.n_visits += 1

            if 4 >= self.ab_p2 >= 3 and i > 1 and isfinite(alpha_val) and isfinite(alpha_val):
                parent: Node = selected[i - 2]

                alpha = beta_val if parent.player == self.player else 1 - beta_val
                beta = alpha_val if parent.player == self.player else 1 - alpha_val

                p_idx: cython.short = self.player - 1
                opp_idx: cython.short = (3 - self.player) - 1
                # Update the node with alpha and beta values to confirm the bounds
                if self.ab_p2 == 3:
                    node.v[p_idx] -= alpha
                    node.v[opp_idx] += beta
                elif self.ab_p2 == 4:
                    node.v[p_idx] += alpha
                    node.v[opp_idx] -= beta

            # Update RAVE statistics for all child nodes, don't update for draws, only win/losses
            if self.rave and not is_draw and node.n_children > 0:
                for j in range(node.n_children):
                    child: Node = node.children[j]
                    # Children are added left to right in expand, so we can break if we reach a None child
                    if child is None:
                        break

                    action: cython.tuple = child.action  # Assumes each child node knows the action that led to it
                    action_list: cython.list[cython.tuple] = p1_actions if node.player == 1 else p2_actions
                    if action in action_list:
                        child.amaf_visits += 1
                        res: cython.float = result[node.player - 1]
                        child.amaf_wins += res

            # * Backpropagate the result of the playout
            if self.imm and node.expanded:
                if node.player == self.player:
                    node.im_value = loss - 1  # Initialize to min negative
                    for j in range(node.n_children):
                        child_im: cython.float = node.children[j].im_value
                        node.im_value = max(node.im_value, child_im)
                else:
                    node.im_value = win + 1  # Initialize to max positive
                    for j in range(node.n_children):
                        child_im: cython.float = node.children[j].im_value
                        node.im_value = min(node.im_value, child_im)

    @cython.cfunc
    @cython.inline
    @cython.locals(i=cython.int, action=cython.tuple)
    def update_player_mast_values(
        self,
        actions: cython.list[cython.tuple],
        player_i: cython.short,
        score: cython.float,
    ) -> cython.void:
        for i in range(len(actions)):
            action = actions[i]
            action_hash: cython.uint = hash_tuple(action, 10000)
            self.mast_values[player_i][action_hash] += score  # Update based on game result
            self.mast_visits[player_i][action_hash] += 1

    @cython.cfunc
    def play_out(
        self,
        state: GameState,
        p1_actions: cython.list[cython.tuple],
        p2_actions: cython.list[cython.tuple],
    ) -> cython.tuple[cython.float, cython.float]:
        turns: cython.short = 0
        best_action: cython.tuple

        while not state.is_terminal():
            if turns > 1:
                # Early termination condition with a fixed number of turns
                if self.early_term and turns >= self.early_term_turns:
                    # ! This assumes symmetric evaluation functions centered around 0!
                    evaluation: cython.float = state.evaluate(params=self.eval_params, player=1, norm=True)
                    if evaluation >= self.early_term_cutoff:
                        return (1.0, 0.0)
                    elif evaluation <= -self.early_term_cutoff:
                        return (0.0, 1.0)
                    else:
                        return (0.5, 0.5)

                # Dynamic Early termination condition, check every two turns if the evaluation has a certain value
                elif self.early_term == 0 and self.dyn_early_term == 1 and turns % 2 == 0:
                    # ! This assumes symmetric evaluation functions centered around 0!
                    evaluation = state.evaluate(params=self.eval_params, player=1, norm=True)
                    if evaluation >= self.dyn_early_term_cutoff:
                        return (1.0, 0.0)
                    elif evaluation <= -self.dyn_early_term_cutoff:
                        return (0.0, 1.0)

            action_selected: cython.bint = False
            if not action_selected and self.mast:
                if uniform(0, 1) < self.epsilon:
                    # Get the legal actions for the current state
                    actions: cython.list[cython.tuple] = state.get_legal_actions()
                    mast_max: cython.float = -INFINITY
                    c_i: cython.short
                    action: cython.tuple
                    # Select the move with the highest MAST value
                    for c_i in range(len(actions)):
                        action = actions[c_i]
                        if action == None:  # This means that the game is over (MiniShogi)
                            action_selected = True
                            best_action = None
                            break
                        action_hash: cython.uint = hash_tuple(action, 10000)

                        mast_vis: cython.float = cython.cast(
                            cython.float, self.mast_visits[state.player - 1][action_hash]
                        )

                        if mast_vis >= 5:
                            mast_val: cython.float = self.mast_values[state.player - 1][action_hash] / mast_vis
                            mast_val += uniform(-0.01, 0.01)  # Add a bit of randome sugah

                            if __debug__:  # Keep track of the number of MAST moves selected
                                if not isfinite(mast_max):
                                    self.mast_moves += (
                                        1  # We know that we will select a MAST move and count it only once.
                                    )
                        else:
                            # We must have tried all the moves at least a few times to get an average
                            mast_val: cython.float = 1.0 + uniform(0, 1)

                        # If bigger, we have a winner, chose this move
                        if mast_val > mast_max:
                            mast_max = mast_val
                            best_action = action
                            action_selected = True

            # With probability epsilon choose the best action from a subset of moves
            if not action_selected and self.e_greedy:
                if uniform(0, 1) < self.epsilon:
                    actions: cython.list[cython.tuple] = state.get_legal_actions()
                    random.shuffle(actions)
                    actions = actions[: self.e_g_subset]
                    max_value: cython.float = -99999.99

                    for i in range(len(actions)):
                        action: cython.tuple = actions[i]
                        if action == None:  # This means that the game is over (MiniShogi)
                            action_selected = True
                            best_action = None
                            break

                        # Evaluate the new state in view of the player to move
                        value: cython.float = state.apply_action(action).evaluate(
                            params=self.eval_params,
                            player=state.player,
                            norm=False,
                        )

                        # Keep track of the best action
                        if value > max_value:
                            max_value = value
                            best_action = actions[i]
                            action_selected = True

            # No mast or e-greedy move was selected, hence select one at random
            if not action_selected:
                best_action = state.get_random_action()
                # For MAST, keep track of the actions taken
                if self.mast:
                    if state.player == 1 and best_action != None:  # ! The player that is GOING TO MAKE the move
                        p1_actions.append(best_action)
                    elif best_action != None:
                        p2_actions.append(best_action)

            state.apply_action_playout(best_action)

            self.avg_po_moves += 1
            turns += 1

        # We've ended the playout in a terminal state, keep track of the number of terminal playouts
        if __debug__:
            self.playout_terminals += 1

        # Map the result to the players
        res_tuple: cython.tuple[cython.float, cython.float] = state.get_result_tuple()
        # multiply all 1's by 1.3 to reward true wins
        if res_tuple[0] == 1:
            return (1.2, 0.0)
        elif res_tuple[1] == 1:
            return (0.0, 1.2)
        else:  # It's a draw
            return (0.5, 0.5)

    def print_cumulative_statistics(self) -> str:
        return ""

    def print_debug_info(self, i: cython.int, total_time: cython.long, max_node: Node, state: GameState):
        print(
            f"\n\n** ran {i+1:,} simulations in {format_time(total_time)}, {i / float(max(1, total_time)):,.0f} simulations per second **"
        )
        if self.root.solved_player != 0:
            print(f"*** Root node is solved for player {self.root.solved_player} ***")
        if self.root.anti_decisive:
            print("*** Anti-decisive move found ***")
        if self.root.draw:
            print("*** Proven draw ***")

        print("--*--" * 20)
        print(f"BEST NODE: {max_node}")
        print("-*=*-" * 15)
        next_state: GameState = state.apply_action(max_node.action)
        evaluation: cython.float = next_state.evaluate(params=self.eval_params, player=self.player, norm=False)
        norm_eval: cython.float = next_state.evaluate(params=self.eval_params, player=self.player, norm=True)
        self.max_eval = max(evaluation, self.max_eval)

        print(
            f"evaluation: {evaluation:.2f} / (normalized): {norm_eval:.4f} | max_eval: {self.max_eval:.1f} | Playout draws: {self.playout_draws:,} | Terminal Playout: {self.playout_terminals:,} | MAST moves: {self.mast_moves:,}"
        )
        print(
            f"max depth: {self.max_depth} | avg depth: {self.avg_depth:.2f}| avg. playout moves: {self.avg_po_moves:.2f} | avg. playouts p/s.: {self.avg_pos_ps / self.n_moves:,.0f} | {self.n_moves} moves played"
        )
        self.avg_po_moves = self.mast_moves = 0
        global ab_bound, ucb_bound
        print(
            f"alpha/beta bound used: {ab_bound:,} |  ucb bound used: {ucb_bound:,} | percentage: {(float(ab_bound) / max(ab_bound + ucb_bound, 1)) * 100:.2f}%"
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

        if __debug__ and self.ab_p1 != 0:

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

    def __repr__(self):
        full_str: str = f"MCTS(p={self.player}"

        if self.name != "":
            full_str += f", {self.name}"

        full_str += f", c={self.c:.2f}"

        if self.move_selection == 0:
            full_str += ", VISITS"
        elif self.move_selection == 1:
            full_str += ", VALUE"

        if self.num_simulations:
            full_str += f", num_sim={self.num_simulations}"
        elif self.max_time:
            full_str += f", max_time={self.max_time}"
        if self.e_greedy:
            full_str += f", eps_subset={self.e_g_subset}, eps={self.epsilon:.3f}"
        if self.rave:
            full_str += f", rave_k={self.rave_k:.2f}"
        if self.mast:
            full_str += f", mast, eps={self.epsilon:.3f}"
        if self.early_term:
            full_str += f", et_turns={self.early_term_turns}, et_cut={self.early_term_cutoff:.2f}"
        if self.imm:
            full_str += f", imm_alpha={self.imm_alpha:.2f}"
        if self.ab_p1 != 0:
            full_str += f", ab_p={self.ab_p1}/{self.ab_p2}, k_factor={self.k_factor:.2f}"

        return full_str + ")"


class ChildComparator:
    def __call__(self, child):
        return child.n_visits


# Define a key function for sorting
def get_node_visits(node):
    return node.n_visits


def plot_selected_bins(bins_dict, plot_width=140, plot_height=32):
    global just_play
    if just_play:
        return
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
        elif user_input == "j":
            just_play = True
            break
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


# @cython.cfunc
# @cython.returns(cython.float)
# @cython.exceptval(-88888888, check=False)  # Don't use 99999 because it's win/los
# @cython.locals(
#     is_max_player=cython.bint,
#     v=cython.float,
#     new_v=cython.float,
#     actions=cython.list,
#     move=cython.tuple,
#     m=cython.short,
#     i=cython.short,
#     n_actions=cython.short,
# )
# def alpha_beta(
#     state: GameState,
#     alpha: cython.float,
#     beta: cython.float,
#     depth: cython.short,
#     max_player: cython.short,
#     eval_params: cython.float[:],
# ):
#     if state.is_terminal():
#         return state.get_reward(max_player)

#     if depth == 0:
#         return state.evaluate(params=eval_params, player=max_player, norm=True) + uniform(-0.0001, 0.0001)

#     is_max_player = state.player == max_player
#     # Move ordering
#     actions: cython.list[cython.tuple] = state.get_legal_actions()
#     # actions = state.evaluate_moves(state.get_legal_actions())
#     # actions = [(actions[i][0], actions[i][1]) for i in range(n_actions)] # ? I don't think that this line is needed here
#     # actions.sort(key=itemgetter(1), reverse=is_max_player)
#     # actions = [actions[i][0] for i in range(n_actions)] # ? Nor is this one needed here as long as we use actions[m][0] in the loop (instead of actions[m])

#     v = -INFINITY if is_max_player else INFINITY
#     for m in range(len(actions)):
#         move = actions[m]

#         new_v = alpha_beta(
#             state=state.apply_action(move),
#             alpha=alpha,
#             beta=beta,
#             depth=depth - 1,
#             max_player=max_player,
#             eval_params=eval_params,
#         )
#         # Update v, alpha or beta based on the player
#         if (is_max_player and new_v > v) or (not is_max_player and new_v < v):
#             v = new_v
#         # Alpha-beta pruning
#         if is_max_player:
#             alpha = max(alpha, new_v)
#         else:
#             beta = min(beta, new_v)
#         # Prune the branch
#         if beta <= alpha:
#             break

#     return v


# @cython.cfunc
# @cython.returns(cython.float)
# @cython.locals(
#     actions=cython.list[cython.tuple],
#     move=cython.tuple,
#     score=cython.float,
#     stand_pat=cython.float,
#     m=cython.short,
# )
# @cython.exceptval(-88888888, check=False)
# def quiescence(
#     state: GameState,
#     stand_pat: cython.float,
#     max_player: cython.short,
#     eval_params: cython.float[:],
# ):
#     actions = state.get_legal_actions()
#     for m in range(len(actions)):
#         move = actions[m]
#         if state.is_capture(move):
#             new_state = state.apply_action(move)
#             new_stand_pat = new_state.evaluate(params=eval_params, player=max_player, norm=True) + uniform(
#                 -0.0001, 0.0001
#             )
#             score = -quiescence(new_state, new_stand_pat, max_player, eval_params)
#             stand_pat = max(stand_pat, score)  # Update best_score if a better score is found

#     return stand_pat  # Return the best score found


# if __debug__:
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
#     "ab_uct_bins": {"bin": DynamicBin(n_bins), "label": "Best UCB values"},
#     "2_dist_bins": {"bin": DynamicBin(n_bins), "label": "Difference between Beta and Alpha for Player 2"},
# }
