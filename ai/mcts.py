# cython: language_level=3

import random
from colorama import Back, Fore, init, Style

init(autoreset=True)

import cython

from cython.cimports.includes import c_uniform_random, c_random, c_shuffle
from cython.cimports.includes.c_util import c_random_seed
from cython.cimports.libc.time import time
from cython.cimports.libc.math import sqrt, log
from cython.cimports.includes import GameState, win, loss

from util import abbreviate, format_time

DEBUG: cython.bint = 0

# TODO Compiler directives toevoegen na debuggen
# TODO After testing, remove assert statements


@cython.cfunc
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

    eval_value: cython.double
    action: cython.tuple
    actions: cython.list  # A list of actions that is used to expand the node

    def __cinit__(self, player: cython.int, action: cython.tuple, max_player: cython.int):
        # The action that led to this state, needed for root selection
        self.action = action
        self.player = player
        self.max_player = max_player
        # Since evaluations are stored in view of player 1, player 1 is the maximizing player
        self.im_value = -99999999.9 if self.player == 1 else 99999999.9
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
        # Just to make sure that there's always a child to select
        max_child: Node = self.children[c_random(0, n_children - 1)]
        max_val: cython.double = -999999.99
        children_lost: cython.int = 0
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
                + c_uniform_random(0.000001, 0.00001)
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

        return max_child  # A random child was chosen at the beginning of the method, hence this will never be None

    @cython.cfunc
    def expand(
        self,
        init_state: GameState,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
    ) -> Node:
        assert not self.expanded, f"Trying to re-expand an already expanded node, you madman. {str(self)}"

        if self.children == []:
            # * Idee: move ordering!
            self.actions = init_state.get_legal_actions()
            self.children = [None] * len(self.actions)

        while len(self.actions) > 0:
            # Get a random action from the list of previously generated actions
            action: cython.tuple = self.actions.pop(c_random(0, len(self.actions) - 1))

            new_state: GameState = init_state.apply_action(action)
            child: Node = Node(new_state.player, action, self.max_player)

            # This works because we pop the actions
            self.children[len(self.children) - len(self.actions) - 1] = child

            # Solver mechanics..
            if new_state.is_terminal():
                result: cython.int = new_state.get_reward(self.player)

                if result == win:  # We've found a winning position, hence this node and the child are solved
                    self.solved_player = child.solved_player = self.player
                # We've found a losing position, we cannot yet tell if it is solved, but no need to return the child
                elif result == loss:
                    child.solved_player = 3 - self.player
                    continue
                # If the game is a draw or a win, we can return the child as any other.
                return child  # Return a child that is not a loss. This is the node we will explore next.
            # No use to evaluate terminal or solved nodes.
            elif prog_bias or imm:
                # * eval_value is always in view of max_player
                eval_value: cython.double = new_state.evaluate(
                    player=self.max_player,
                    norm=True,
                    params=eval_params,
                )

                child.eval_value = eval_value
                if imm:
                    # At first, the im value of the child is the same as the evaluation value, when searching deeper, it becomes the min/max of the subtree
                    child.im_value = eval_value
                    # This means that player 2 is minimizing the evaluated value and player 1 is maximizing it
                    if (self.player != self.max_player and eval_value < self.im_value) or (
                        self.player == self.max_player and eval_value > self.im_value
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

        if imm:
            # Do the full minimax back-up
            best_im: cython.double
            if self.player == self.max_player:
                best_im = -99999999.9
            else:
                best_im = 99999999.9

            i: cython.int
            for i in range(len(self.children)):
                child = self.children[i]
                if self.player == self.max_player:  # * Maximize my im value over all children
                    self.im_value = max(best_im, child.im_value)
                else:  # * Minimize my im value over all children
                    self.im_value = min(best_im, child.im_value)

        # If we've reached this point, then the node is fully expanded so we can switch to UCT selection
        self.expanded = 1
        # Check if all my nodes lead to a loss.
        self.check_loss_node()

    @cython.cfunc
    def add_all_children(
        self,
        init_state: GameState,
        eval_params: cython.double[:],
        prog_bias: cython.bint = 0,
        imm: cython.bint = 0,
    ):
        """
        Adds a previously expanded node's children to the tree
        """
        while not self.expanded:
            self.expand(init_state, prog_bias=prog_bias, imm=imm, eval_params=eval_params)

    @cython.cfunc
    def check_loss_node(self):
        # I'm solved, nothing to see here.
        if self.solved_player != 0.0:
            return

        opponent: cython.int = 3 - self.player
        child: Node
        i: cython.int

        for i in range(len(self.children)):
            child = self.children[i]
            # If all children lead to a loss, then we will not return from the function
            if child.solved_player != opponent:
                return

        self.solved_player = opponent

    def __str__(self):
        root_mark = "(Root)" if self.action == () else ""

        solved_bg = ""
        if self.solved_player == 1:
            solved_bg = Back.LIGHTCYAN_EX
        elif self.solved_player == 2:
            solved_bg = Back.LIGHTWHITE_EX

        im_str = (
            f"{Fore.RED}IM:{Style.BRIGHT}{self.im_value:7.2f}{Style.NORMAL} "
            if abs(self.im_value) != 99999999.9
            else ""
        )

        # This is flipped because I want to see it in view of the parent
        value = (self.v[(3 - self.player) - 1] - self.v[self.player - 1]) / self.n_visits

        return (
            f"{solved_bg}"
            f"{Fore.BLUE}A:{Style.BRIGHT}{str(self.action):<10}{Style.NORMAL}{root_mark} "
            f"{Fore.GREEN}P:{Style.BRIGHT}{self.player:<3}{Style.NORMAL} "
            + im_str
            + f"{Fore.YELLOW}EV:{Style.BRIGHT}{self.eval_value:7.2f}{Style.NORMAL} "
            f"{Fore.CYAN}EX:{Style.BRIGHT}{self.expanded:<3}{Style.NORMAL} "
            f"{Fore.MAGENTA}SP:{Style.BRIGHT}{self.solved_player:<3}{Style.NORMAL} "
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
    dyn_early_term_cutoff: cython.double
    c: cython.double
    epsilon: cython.double
    max_time: cython.double
    eval_params: cython.double[:]
    root: Node

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
        e_g_subset: int = 20,
        imm_alpha: float = 0.4,
        imm: bool = False,
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

    @cython.ccall
    def best_action(self, state: GameState) -> cython.tuple:
        assert state.player == self.player, "The player to move does not match my max player"

        # Check if we can reutilize the root
        # If the root is None then this is either the first move, or something else..
        if self.root is not None:
            child: Node
            children: cython.list = self.root.children
            self.root = None  # In case we cannot find the action, mark the root as None to assert

            for child in children:
                if child is None:
                    break
                if child.action == state.last_action:
                    self.root = child
                    if DEBUG:
                        print("Reusing root node")
                        self.root.action = ()  # This is what identifies the root node
                    break
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
                        print(f"\rSimulation: {i+1} ", end="")

                self.simulate(state)
        else:
            while curr_time() - start_time < self.max_time:
                if DEBUG:
                    if (i + 1) % 100 == 0:
                        print(f"\rSimulation: {i+1} ", end="")

                self.simulate(state)
                i += 1

        if DEBUG:
            total_time: cython.long = curr_time() - start_time
            print(
                f"Ran {i+1:,} simulations in {format_time(total_time)}, {i / float(max(1, total_time)):,.0f} simulations per second."
            )

        # retrieve the node with the most visits
        assert (
            len(self.root.children) > 0
        ), f"No children found for root node {self.root}, after {i:,} simulations"

        max_node: Node = self.root.children[0]
        max_value: cython.double = max_node.n_visits
        n_children: cython.int = len(self.root.children)

        # TODO Idea, use the im_value to decide which child to select
        for i in range(1, n_children):
            # ! A none exception could happen in case there's a mistake in how the children are added to the list in expand
            node: Node = self.root.children[i]
            if node.solved_player == self.player:  # We found a winning move, let's go champ
                max_node = node
                max_value = node.n_visits
                break
            value: cython.double = node.n_visits
            if value > max_value:
                max_node = node
                max_value = value

        if DEBUG:
            print("--*--" * 20)
            print(f"BEST NODE: {max_node}")
            print(
                f"Previous state evaluation: {state.evaluate(params=self.eval_params, player=self.player, norm=False):.4f} / (normalized): {state.evaluate(params=self.eval_params, player=self.player, norm=True):.4f}"
            )
            print("--*--" * 20)

        if DEBUG:
            print(f":: {self.root} :: ")
            print(":: Children ::")
            comparator = ChildComparator()
            sorted_children = sorted(self.root.children[:20], key=comparator, reverse=True)
            print("\n".join([str(child) for child in sorted_children]))

        # For tree reuse, make sure that we can access the next action from the root
        self.root = max_node
        return max_node.action, max_value  # return the most visited state

    @cython.cfunc
    def simulate(self, init_state: GameState):
        # The root is solved, no need to look any further, since we have a winning/losing move
        if self.root.solved_player != 0:
            return

        node: Node = self.root
        selected: cython.list = [self.root]

        next_state: GameState = init_state
        is_terminal: cython.bint = init_state.is_terminal()

        # Select: non-terminal, non-solved, expanded nodes
        while not is_terminal and node.expanded and node.solved_player == 0:
            node = node.uct(self.c, self.pb_weight, self.imm_alpha)
            # TODO If we only make a copy of the first state, then we do not have to make further copies
            next_state = next_state.apply_action(node.action)
            selected.append(node)

            is_terminal = next_state.is_terminal()

        result: tuple[cython.double, cython.double]
        # If the node is neither terminal nor solved, then we need a playout
        if not is_terminal and node.solved_player == 0:
            # Expansion returns the expanded node
            next_node: Node = node.expand(
                init_state=next_state, prog_bias=self.prog_bias, imm=self.imm, eval_params=self.eval_params
            )

            # This is the point where the last action was previously added to the node, so in fact the node is just marked as expanded
            if next_node == None:
                assert node.expanded, "The node should have been expanded"
                next_node = node.uct(self.c, self.pb_weight, self.imm_alpha)

            next_state = next_state.apply_action(next_node.action)
            selected.append(next_node)

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

        i: cython.int
        c: cython.int
        min_im_val: cython.double
        max_im_val: cython.double
        # Backpropagate the result along the chosen nodes
        for i in range(len(selected), 0, -1):  # Move backwards through the list
            node = selected[i - 1]  # In reverse, start is inclusive, stop is exclusive
            if self.imm and node.expanded:  # The last node selected (first in the loop) is not expanded
                # Update the im_values of the node based on min/maxing the im_values of the children
                if node.player == self.player:  # maximize im_value
                    max_im_val = -99999999.9
                    for c in range(len(node.children)):
                        # ! none exception if expand is not properly working
                        max_im_val = max(max_im_val, node.children[c].im_value)
                    node.im_value = max_im_val

                else:  # minimize im_value
                    min_im_val = 99999999.9
                    for c in range(len(node.children)):
                        # ! none exception if expand is not properly working
                        min_im_val = min(min_im_val, node.children[c].im_value)
                    node.im_value = min_im_val

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
            if self.dyn_early_term == 1 and turns % 5 == 0:
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
                c_shuffle(actions)

                max_value = -99999.99
                for i in range(self.e_g_subset):
                    # Evaluate the new state in view of the player to move
                    value = state.apply_action(actions[i]).evaluate(
                        params=self.eval_params,
                        player=state.player,
                        norm=False,
                    )
                    if value > max_value:
                        max_value = value
                        best_action = actions[i]

            # With probability epsilon choose a move using roulette wheel selection based on the move ordering
            elif self.roulette == 1 and c_uniform_random(0, 1) < self.epsilon:
                actions = state.get_legal_actions()
                best_action = random.choices(actions, weights=state.move_weights(actions), k=1)[0]

            # With probability 1-epsilon chose a move at random
            if best_action == ():
                best_action = state.get_random_action()

            state.apply_action_playout(best_action)

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
