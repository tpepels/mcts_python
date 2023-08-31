# cython: language_level=3

from operator import itemgetter
import random
from colorama import Back, Fore, init, Style

init(autoreset=True)

import cython

from cython.cimports.includes import c_uniform_random, c_random, c_shuffle
from cython.cimports.includes.c_util import c_random_seed
from cython.cimports.libc.time import time
from cython.cimports.libc.math import sqrt, log
from cython.cimports.includes import GameState, win, loss
from cython.cimports.libc.math import INFINITY

from util import abbreviate, format_time

DEBUG: cython.bint = 0

# TODO Compiler directives toevoegen na debuggen
# TODO After testing, remove assert statements


@cython.cfunc
@cython.exceptval(-1, check=False)
def curr_time() -> cython.long:
    return time(cython.NULL)


c_random_seed(curr_time())


@cython.freelist(10000)
@cython.cclass
class Node:
    children = cython.declare(cython.list, visibility="public")
    state_hash = cython.declare(cython.longlong, visibility="public")
    # State values
    v = cython.declare(cython.double[2], visibility="public")

    n_visits = cython.declare(cython.int, visibility="public")
    expanded = cython.declare(cython.bint, visibility="public")
    im_value = cython.declare(cython.double, visibility="public")
    solved_player = cython.declare(cython.int, visibility="public")
    player = cython.declare(cython.int, visibility="public")
    max_player = cython.declare(cython.int, visibility="public")
    anti_decisive = cython.declare(cython.bint, visibility="public")

    draw: cython.bint
    eval_value: cython.double
    action: cython.tuple
    actions: cython.list  # A list of actions that is used to expand the node

    def __cinit__(self, player: cython.int, action: cython.tuple, max_player: cython.int):
        # The action that led to this state, needed for root selection
        self.action = action
        self.player = player
        self.max_player = max_player
        # Now we know whether we are minimizing or maximizing
        self.im_value = float("-inf") if self.player == self.max_player else float("inf")
        self.im_value = 0
        self.eval_value = 0.0
        self.expanded = 0
        self.solved_player = 0
        self.v = [0.0, 0.0]  # Values for player-1
        self.n_visits = 0
        self.actions = []
        self.children = []

    @cython.cfunc
    def uct(self, c: cython.double, pb_weight: cython.double = 0.0, imm_alpha: cython.double = 0.0) -> Node:
        n_children: cython.int = len(self.children)
        assert n_children > 0, "Trying to uct a node without children"
        assert self.expanded, "Trying to uct a node that is not expanded"
        # Just to make sure that there's always a child to select
        max_child: Node = self.children[c_random(0, n_children - 1)]
        max_val: cython.double = -999999.99
        children_lost: cython.int = 0
        children_draw: cython.int = 0
        ci: cython.int
        # Move through the children to find the one with the highest UCT value
        for ci in range(n_children):
            child: Node = self.children[ci]
            # ! A none exception could happen in case there's a mistake in how the children are added to the list in expand
            # Make sure that every child is seen at least once.
            if child.n_visits == 0:
                return child

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

            avg_value: cython.double = (
                child.v[self.player - 1] - child.v[(3 - self.player) - 1]
            ) / child.n_visits

            # Implicit minimax
            if imm_alpha > 0.0:
                if self.player == self.max_player:  # Maximize the im value
                    avg_value = ((1.0 - imm_alpha) * avg_value) + (imm_alpha * child.im_value)
                else:
                    # Maximize the negative im value
                    avg_value = ((1.0 - imm_alpha) * avg_value) + (imm_alpha * -child.im_value)

            pb_h: cython.double
            if self.player == self.max_player:
                pb_h = child.eval_value
            else:
                pb_h = -child.eval_value

            uct_val: cython.double = (
                avg_value
                + sqrt(c * log(self.n_visits) / child.n_visits)
                # + c_uniform_random(0.000001, 0.00001) # This was taking more time than expected
                # Progressive bias, set pb_weight to 0 to disable
                + (pb_weight * (pb_h / (1.0 + child.n_visits)))
            )

            if uct_val >= max_val:
                max_child = child
                max_val = uct_val

        # It may happen that somewhere else in the tree all my children have been found to be proven losses.
        if children_lost == n_children:
            # Since all children are losses, we can mark this node as solved for the opponent
            # We do this here because the node may have been expanded and solved elsewhere in the tree
            self.solved_player = 3 - self.player
        elif children_lost == (n_children - 1) and self.solved_player == 0:
            # There's only one move that does not lead to a loss. This is an anti-decisive move.
            self.anti_decisive = 1
        # Proven draw
        elif children_draw == n_children:
            self.draw = 1

        return max_child  # A random child was chosen at the beginning of the method, hence this will never be None

    # TODO Misschien moet je een versie maken die een list van nodes teruggeeft zodat je dieper dan 1 laag kan expanden?
    # TODO Of de IMM versie 3 in simulate doen? En daar de acties bijhouden
    # TODO Of naast de node ook een nieuwe state meegeven?
    # TODO Eigenlijk moet het gewoon recursief, in plaats van met de "handmatige" stack
    @cython.cfunc
    def expand(
        self,
        init_state: GameState,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
        imm_version: cython.int = 0,
        imm_ex_D: cython.int = 0,
    ) -> Node:
        assert not self.expanded, f"Trying to re-expand an already expanded node, you madman. {str(self)}"

        if self.children == []:
            # * Idee: move ordering!
            self.actions = init_state.get_legal_actions()
            self.children = [None] * len(self.actions)

        while len(self.actions) > 0:
            # Get a random action from the list of previously generated actions
            action: cython.tuple = self.actions.pop()

            new_state: GameState = init_state.apply_action(action)
            child: Node = Node(new_state.player, action, self.max_player)

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
                    eval_value: cython.double = new_state.evaluate(
                        player=self.max_player,
                        norm=True,
                        params=eval_params,
                    )
                    child.eval_value = eval_value
                if imm:
                    # At first, the im value of the child is the same as the evaluation value, when searching deeper, it becomes the min/max of the subtree
                    if (
                        imm_version == 0 or imm_version == 3
                    ):  # Plain old vanilla imm, just set the evaluation score
                        # TODO If prog_bias is enabled, then we are computing the same thing twice
                        child.im_value = new_state.evaluate(
                            player=self.max_player,
                            norm=True,
                            params=eval_params,
                        )
                    elif imm_version == 1 or imm_version == 13:  # n-ply-imm, set the evaluation score
                        child.im_value = alpha_beta(
                            state=new_state,
                            eval_params=eval_params,
                            max_player=self.max_player,
                            depth=imm_ex_D,  # This is the depth that we will search
                            alpha=-float("inf"),  # ? Can we set these to some meaningful values?
                            beta=float("inf"),  # ? Can we set these to some more meaningful values?
                        )

                    elif (
                        imm_version == 2 or imm_version == 23
                    ):  # q-imm, set the evaluation score to the quiescence score
                        # TODO If prog_bias is enabled, then we are computing the same thing twice
                        eval_value: cython.double = new_state.evaluate(
                            player=self.max_player,
                            norm=True,
                            params=eval_params,
                        )
                        # Play out capture sequences
                        if init_state.is_capture(action):
                            child.im_value = quiescence(
                                state=new_state,
                                stand_pat=eval_value,
                                eval_params=eval_params,
                                max_player=self.max_player,
                            )
                        else:
                            child.im_value = eval_value

                    elif imm_version == 12:  # n-ply-imm with quiescence
                        # Play out capture sequences
                        if init_state.is_capture(action):
                            # TODO If prog_bias is enabled, then we are computing the same thing twice
                            eval_value: cython.double = new_state.evaluate(
                                player=self.max_player,
                                norm=True,
                                params=eval_params,
                            )
                            child.im_value = quiescence(
                                state=new_state,
                                stand_pat=eval_value,
                                eval_params=eval_params,
                                max_player=self.max_player,
                            )
                        else:
                            child.im_value = alpha_beta(
                                state=new_state,
                                eval_params=eval_params,
                                max_player=self.max_player,
                                depth=imm_ex_D,  # This is the depth that we will search
                                alpha=-float("inf"),  # ? Can we set these to some meaningful values?
                                beta=float("inf"),  # ? Can we set these to some more meaningful values?
                            )

                    # This means that player 2 is minimizing the evaluated value and player 1 is maximizing it
                    if (self.player != self.max_player and child.im_value < self.im_value) or (
                        self.player == self.max_player and child.im_value > self.im_value
                    ):
                        # The child improves the current evaluation, hence we can update the evaluation value
                        self.im_value = child.im_value
                    elif imm_version == 3 or imm_version == 13 or imm_version == 23:
                        # Give each child a visit, to make sure the imm score is used in the uct selection and no unnessesary nodes are added
                        child.n_visits += 1
                        # Keep adding nodes until we find a node that improves the im value, i.e. don't return just yet
                        continue

            return child  # Return the chlid, this is the node we will explore next.
        # If we've reached this point, then the node is fully expanded so we can switch to UCT selection
        self.expanded = 1
        # Check if all my nodes lead to a loss.
        self.check_loss_node()

        if imm:
            # Do the full minimax back-up
            best_im: cython.double
            best_node: Node

            best_im = -INFINITY if self.player == self.max_player else INFINITY

            i: cython.int
            for i in range(len(self.children)):
                child = self.children[i]
                if (self.player == self.max_player and child.im_value > best_im) or (
                    self.player != self.max_player and child.im_value < best_im
                ):
                    # * Minimize or Maximize my im value over all children
                    best_im = child.im_value
                    best_node = child

            self.im_value = best_im
            # Return the best node for another visit
            return best_node

        # Return a random child for another visit
        return self.children[c_random(0, len(self.children) - 1)]

    @cython.cfunc
    def add_all_children(
        self,
        init_state: GameState,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
        imm_version: cython.int = 0,
        imm_ex_D: cython.int = 0,
    ):
        """
        Adds a previously expanded node's children to the tree
        """
        while not self.expanded:
            self.expand(
                init_state,
                prog_bias=prog_bias,
                imm=imm,
                eval_params=eval_params,
                imm_version=imm_version,
                imm_ex_D=imm_ex_D,
            )

    @cython.cfunc
    @cython.locals(child=Node, i=cython.int)
    def check_loss_node(self):
        # I'm solved, nothing to see here.
        if self.solved_player != 0.0:
            return

        for i in range(len(self.children)):
            child = self.children[i]
            # If all children lead to a loss, then we will not return from the function
            if child.solved_player != 3 - self.player:
                return

        self.solved_player = 3 - self.player

    def __str__(self):
        root_mark = f"(Root) AD:{self.anti_decisive:<1}" if self.action == () else ""

        solved_bg = ""
        if self.solved_player == 1:
            solved_bg = Back.LIGHTCYAN_EX
        elif self.solved_player == 2:
            solved_bg = Back.LIGHTWHITE_EX
        elif self.draw:
            solved_bg = Back.LIGHTGREEN_EX

        im_str = (
            f"{Fore.RED}IM:{Style.BRIGHT}{self.im_value:7.2f}{Style.NORMAL} "
            if abs(self.im_value) != float("inf")
            else ""
        )

        # This is flipped because I want to see it in view of the parent
        value = (self.v[(3 - self.player) - 1] - self.v[self.player - 1]) / max(1, self.n_visits)

        return (
            f"{solved_bg}"
            f"{Fore.BLUE}A:{Style.BRIGHT}{str(self.action):<10}{Style.NORMAL}{root_mark} "
            f"{Fore.GREEN}P:{Style.BRIGHT}{self.player:<3}{Style.NORMAL} "
            + im_str
            + f"{Fore.YELLOW}EV:{Style.BRIGHT}{self.eval_value:7.2f}{Style.NORMAL} "
            f"{Fore.CYAN}EX:{Style.BRIGHT}{self.expanded:<3}{Style.NORMAL} "
            f"{Fore.MAGENTA}SP:{Style.BRIGHT}{self.solved_player:<3}{Style.NORMAL} "
            f"{Fore.MAGENTA}DR:{Style.BRIGHT}{self.draw:<3}{Style.NORMAL} "
            f"{Fore.WHITE}V:{Style.BRIGHT}{value:2.1f}{Style.NORMAL} "
            f"{Back.YELLOW + Fore.BLACK}NV:{Style.BRIGHT}{self.n_visits}{Style.NORMAL}{Back.RESET + Fore.RESET}"
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
    prog_bias: cython.bint
    pb_weight: cython.double
    imm: cython.bint
    imm_alpha: cython.double
    debug: cython.bint
    roulette: cython.bint
    roulette_eps: cython.double
    dyn_early_term_cutoff: cython.double
    c: cython.double
    epsilon: cython.double
    max_time: cython.double
    eval_params: cython.double[:]
    root: Node
    # The highest evaluation value seen throughout the game (for normalisation purposes later on)
    max_eval: cython.double
    # The average depth of the playouts, for science
    avg_po_moves: cython.double
    # The average number of playouts per second
    avg_pos_ps: cython.double
    n_moves: cython.int
    max_depth: cython.int
    avg_depth: cython.int
    # Research additions
    imm_version: cython.int
    ex_imm_D: cython.int  # The extra depth that will be searched

    def __init__(
        self,
        player: int,
        eval_params: cython.double[:],
        transposition_table_size: int = 2**16,  # This parameter is unused but is passed by run_game
        num_simulations: int = 0,
        max_time: int = 0,
        c: float = 1.0,
        dyn_early_term: bool = False,
        dyn_early_term_cutoff: float = 0.9,
        early_term: bool = False,
        early_term_turns: int = 10,
        early_term_cutoff: float = 0.05,
        e_greedy: bool = False,
        roulette_epsilon: float = 0.05,
        e_g_subset: int = 20,
        imm_alpha: float = 0.4,
        imm: bool = False,
        imm_version: int = 0,
        ex_imm_D: int = 2,
        roulette: bool = False,
        epsilon: float = 0.05,
        prog_bias: bool = False,
        pb_weight: float = 0.5,
        debug: bool = False,
    ):
        self.player = player
        self.dyn_early_term = dyn_early_term
        self.dyn_early_term_cutoff = dyn_early_term_cutoff
        self.early_term_cutoff = early_term_cutoff
        self.e_greedy = e_greedy
        self.epsilon = epsilon
        self.roulette_eps = roulette_epsilon
        self.e_g_subset = e_g_subset
        self.early_term = early_term
        self.early_term_turns = early_term_turns
        self.roulette = roulette
        self.c = c
        self.eval_params = eval_params

        self.imm = imm
        if self.imm:
            self.imm_alpha = imm_alpha
        else:
            self.imm_alpha = 0.0

        self.prog_bias = prog_bias
        if self.prog_bias:
            self.pb_weight = pb_weight
        else:
            self.pb_weight = 0.0

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
        # Research parameters
        self.imm_version = imm_version
        self.ex_imm_D = ex_imm_D

    @cython.ccall
    def best_action(self, state: GameState) -> cython.tuple:
        assert state.player == self.player, "The player to move does not match my max player"
        self.n_moves += 1
        # Check if we can reutilize the root
        # If the root is None then this is either the first move, or something else..
        if state.REUSE and self.root is not None and self.root.expanded:
            child: Node
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

        if self.root is None:
            # Reset root for new round of MCTS
            self.root: Node = Node(state.player, (), self.player)

        if not self.root.expanded:
            self.root.add_all_children(
                init_state=state,
                prog_bias=self.prog_bias,
                imm=self.imm,
                eval_params=self.eval_params,
            )

        start_time: cython.long = curr_time()
        i: cython.int = 0

        if self.num_simulations:
            for i in range(self.num_simulations):
                if DEBUG:
                    if (i + 1) % 100 == 0:
                        print(f"\rSimulation: {i+1:,} ", end="")

                self.simulate(state)
                # The root is solved, no need to look any further, since we have a winning/losing move
                if self.root.solved_player != 0:
                    break
                if self.root.anti_decisive:
                    break
                if self.root.draw:
                    break
        else:
            while curr_time() - start_time < self.max_time:
                if DEBUG:
                    if (i + 1) % 100 == 0:
                        print(f"\rSimulation: {i+1:,} ", end="")

                self.simulate(state)
                i += 1
                # The root is solved, no need to look any further, since we have a winning/losing move
                if self.root.solved_player != 0:
                    break
                if self.root.anti_decisive:
                    break
                if self.root.draw:
                    break

        total_time: cython.long = curr_time() - start_time

        self.avg_po_moves = self.avg_po_moves / (i + 1)
        self.avg_pos_ps += i / float(max(1, total_time))
        self.avg_depth = cython.cast(cython.int, self.avg_depth / (i + 1))

        if DEBUG:
            print(
                f"\n\n** ran {i+1:,} simulations in {format_time(total_time)}, {i / float(max(1, total_time)):,.0f} simulations per second **"
            )
            if self.root.solved_player != 0:
                print(f"*** Root node is solved for player {self.root.solved_player} ***")
            if self.root.anti_decisive:
                print(f"*** Anti-decisive move found ***")
            if self.root.draw:
                print("*** Proven draw ***")

        # retrieve the node with the most visits
        assert (
            len(self.root.children) > 0
        ), f"No children found for root node {self.root}, after {i:,} simulations"

        max_node: Node = None
        max_value: cython.double = -float("inf")
        n_children: cython.int = len(self.root.children)

        c_i: cython.int
        # TODO Idea, use the im_value to decide which child to select
        for c_i in range(0, n_children):
            node: Node = self.root.children[c_i]
            if node.solved_player == self.player:  # We found a winning move, let's go champ
                max_node = node
                max_value = node.n_visits
                break

            if node.solved_player == 3 - self.player:  # We found a losing move
                continue

            value: cython.double = node.n_visits
            if value > max_value:
                max_node = node
                max_value = value

        # In case there's no move that does not lead to a loss, we should return a random move
        if max_node is None:
            if DEBUG:
                print(f"** No winning move found, returning random move **")

            if DEBUG:
                if self.root.solved_player != (3 - self.player):
                    print("Root not solved for opponent and no max_node found!!")
                    print(f"Root node: {str(self.root)}")
                    print("\n".join([str(child) for child in self.root.children]))

            # return a random action if all children are losing moves
            max_node = self.root.children[c_random(0, n_children - 1)]

        if DEBUG:
            print("--*--" * 20)
            print(f"BEST NODE: {max_node}")
            print("-*=*-" * 15)
            next_state: GameState = state.apply_action(max_node.action)
            evaluation: cython.double = next_state.evaluate(
                params=self.eval_params, player=self.player, norm=False
            )
            norm_eval: cython.double = next_state.evaluate(
                params=self.eval_params, player=self.player, norm=True
            )
            self.max_eval = max(evaluation, self.max_eval)
            print(
                f"evaluation: {evaluation:.2f} / (normalized): {norm_eval:.4f} | max_eval: {self.max_eval:.1f}"
            )
            print(
                f"avg. playout moves: {self.avg_po_moves:.2f} | avg. playouts p/s.: {self.avg_pos_ps / self.n_moves:,.0f} | {self.n_moves} moves played"
            )
            self.avg_po_moves = 0
            print(f"max depth: {self.max_depth} | avg depth: {self.avg_depth:.2f}")
            self.max_depth = self.avg_depth = 0
            print("--*--" * 20)

        if DEBUG:
            print(f":: {self.root} :: ")
            print(":: Children ::")
            comparator = ChildComparator()
            sorted_children = sorted(self.root.children[:20], key=comparator, reverse=True)
            print("\n".join([str(child) for child in sorted_children]))
            print("--*--" * 20)

        # For tree reuse, make sure that we can access the next action from the root
        self.root = max_node
        return max_node.action, max_value  # return the most visited state

    @cython.cfunc
    def simulate(self, init_state: GameState):
        # Start at the root
        node: Node = self.root
        # Keep track of selected nodes
        selected: cython.list = [self.root]
        # Keep track of the state
        next_state: GameState = init_state
        is_terminal: cython.bint = init_state.is_terminal()

        expanded: cython.bint = 0  # Ensure expansion only occurs once
        while not is_terminal and node.solved_player == 0:
            if node.expanded:
                node = node.uct(self.c, self.pb_weight, self.imm_alpha)
            elif not node.expanded and not expanded:
                # * Expand should always returns a node, even after adding the last node
                node = node.expand(
                    init_state=next_state,
                    prog_bias=self.prog_bias,
                    imm=self.imm,
                    eval_params=self.eval_params,
                    imm_version=self.imm_version,
                    imm_ex_D=self.ex_imm_D,
                )
                expanded = 1
            elif expanded:
                break

            next_state = next_state.apply_action(node.action)
            is_terminal = next_state.is_terminal()
            selected.append(node)

        # * Playout / Terminal node reached
        result: tuple[cython.double, cython.double]
        # If the node is neither terminal nor solved, then we need a playout
        if not is_terminal and node.solved_player == 0:
            # Do a random playout and collect the result
            result = self.play_out(next_state)
        else:
            if node.solved_player == 0 and is_terminal:
                # A terminal node is reached, so we can backpropagate the result of the state as if it was a playout
                result = next_state.get_result_tuple()
            elif node.solved_player == 1:
                result = (1.0, 0.0)
            elif node.solved_player == 2:
                result = (0.0, 1.0)
            else:
                assert False, "This should not happen!"

        # Keep track of the max depth of the tree
        self.max_depth = max(self.max_depth, len(selected))
        self.avg_depth += len(selected)

        # * Backpropagation
        i: cython.int
        # Backpropagate the result along the chosen nodes
        for i in range(len(selected), 0, -1):  # Move backwards through the list
            node = selected[i - 1]  # In reverse, start is inclusive, stop is exclusive
            # * Imm backpropagation
            if self.imm and node.expanded:
                child_values: cython.list = [child.im_value for child in node.children]
                if node.player == self.player:  # maximize im_value
                    node.im_value = max(child_values, default=-INFINITY)
                else:  # minimize im_value
                    node.im_value = min(child_values, default=INFINITY)
            # * Backpropagate the result of the playout
            node.v[0] += result[0]
            node.v[1] += result[1]
            node.n_visits += 1

    @cython.cfunc
    def play_out(self, state: GameState) -> cython.tuple[cython.double, cython.double]:
        turns: cython.int = 0

        while not state.is_terminal():
            turns += 1
            # Early termination condition with a fixed number of turns
            if self.early_term and turns >= self.early_term_turns:
                # ! This assumes symmetric evaluation functions centered around 0!
                # TODO Figure out the a (max range) for each evaluation function
                evaluation: cython.double = state.evaluate(params=self.eval_params, player=1, norm=True)
                if evaluation > self.early_term_cutoff:
                    return (1.0, 0.0)
                elif evaluation < -self.early_term_cutoff:
                    return (0.0, 1.0)
                else:
                    return (0.5, 0.5)

            # Dynamic Early termination condition, check every 5 turns if the evaluation has a certain value
            if self.dyn_early_term == 1 and turns % 6 == 0:
                # ! This assumes symmetric evaluation functions centered around 0!
                # TODO Figure out the a (max range) for each evaluation function
                evaluation = state.evaluate(params=self.eval_params, player=1, norm=True)

                if evaluation > self.dyn_early_term_cutoff:
                    return (1.0, 0.0)
                elif evaluation < -self.dyn_early_term_cutoff:
                    return (0.0, 1.0)

            best_action: cython.tuple = ()
            # With probability epsilon choose the best action from a subset of moves
            if self.e_greedy == 1 and c_uniform_random(0, 1) < self.epsilon:
                actions = state.get_legal_actions()
                actions = actions[: self.e_g_subset]

                max_value = -99999.99
                for i in range(len(actions)):
                    # Evaluate the new state in view of the player to move
                    value = state.apply_action(actions[i]).evaluate(
                        params=self.eval_params,
                        player=state.player,
                        norm=False,
                    ) + c_uniform_random(0.000001, 0.00001)
                    if value > max_value:
                        max_value = value
                        best_action = actions[i]

            # With probability epsilon choose a move using roulette wheel selection based on the move ordering
            elif self.roulette == 1 and c_uniform_random(0, 1) < self.roulette_eps:
                actions = state.get_legal_actions()
                best_action = random.choices(actions, weights=state.move_weights(actions), k=1)[0]

            # With probability 1-epsilon chose a move at random
            if best_action == ():
                best_action = state.get_random_action()

            state.apply_action_playout(best_action)

        self.avg_po_moves += turns
        # Map the result to the players
        return state.get_result_tuple()

    def print_cumulative_statistics(self) -> str:
        return ""

    def __repr__(self):
        return abbreviate(
            f"MCTS(p={self.player}, eval_params={self.eval_params}, "
            f"num_simulations={self.num_simulations}, "
            f"max_time={self.max_time}, c={self.c}, dyn_early_term={self.dyn_early_term}, "
            f"dyn_early_term_cutoff={self.dyn_early_term_cutoff}, early_term={self.early_term}, "
            f"early_term_turns={self.early_term_turns}, e_greedy={self.e_greedy}, "
            f"e_g_subset={self.e_g_subset}, roulette={self.roulette}, epsilon={self.epsilon}, "
            f"prog_bias={self.prog_bias}, imm={self.imm}, imm_alpha={self.imm_alpha})"
        )


class ChildComparator:
    def __call__(self, child):
        return child.n_visits


@cython.cfunc
@cython.returns(cython.double)
@cython.exceptval(-88888888, check=False)  # Don't use 99999 because it's win/los
@cython.locals(
    is_max_player=cython.bint,
    v=cython.double,
    new_v=cython.double,
    actions=cython.list,
    move=cython.tuple,
    m=cython.int,
    i=cython.int,
    n_actions=cython.int,
)
def alpha_beta(
    state: GameState,
    alpha: cython.double,
    beta: cython.double,
    depth: cython.int,
    max_player: cython.int,
    eval_params: cython.double[:],
):
    if state.is_terminal():
        return state.get_reward(max_player)

    if depth == 0:
        return state.evaluate(params=eval_params, player=max_player, norm=True)

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
@cython.returns(cython.double)
@cython.locals(
    actions=cython.list, move=cython.tuple, score=cython.double, stand_pat=cython.double, m=cython.int
)
@cython.exceptval(-88888888, check=False)
def quiescence(
    state: GameState,
    stand_pat: cython.double,
    max_player: cython.int,
    eval_params: cython.double[:],
):
    actions = state.get_legal_actions()

    for m in range(len(actions)):
        move = actions[m]
        if state.is_capture(move):
            new_state = state.apply_action(move)
            new_stand_pat = new_state.evaluate(params=eval_params, player=max_player, norm=False)
            score = -quiescence(new_state, new_stand_pat, max_player, eval_params)
            stand_pat = max(stand_pat, score)  # Update best_score if a better score is found

    return stand_pat  # Return the best score found
