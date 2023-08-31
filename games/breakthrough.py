# cython: language_level=3
# distutils: language=c++
import array
import cython
import numpy as np

import random

from cython.cimports.libcpp.vector import vector
from cython.cimports.libcpp.pair import pair
from cython.cimports.includes import c_random, normalize, where_is_k, win, loss, draw, GameState
from cython.cimports.games.breakthrough import dirs, lorentz_values, BL_DIR, WH_DIR

dirs = [-1, 0, 1]

# List of values representing the importance of each square on the board. In view of player 2.
lorentz_values = [
    5,
    15,
    15,
    5,
    5,
    15,
    15,
    5,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    4,
    6,
    6,
    6,
    6,
    6,
    6,
    4,
    7,
    10,
    10,
    10,
    10,
    10,
    10,
    7,
    11,
    15,
    15,
    15,
    15,
    15,
    15,
    11,
    16,
    21,
    21,
    21,
    21,
    21,
    21,
    16,
    20,
    28,
    28,
    28,
    28,
    28,
    28,
    20,
    36,
    36,
    36,
    36,
    36,
    36,
    36,
    36,
]


if cython.compiled:
    print("Breakthrough is compiled.")
else:
    print("Breakthrough is just a lowly interpreted script.")


@cython.cclass
class BreakthroughGameState(GameState):
    zobrist_table = [[[random.randint(1, 2**61 - 1) for _ in range(3)] for _ in range(8)] for _ in range(8)]
    REUSE = True

    # player = cython.declare(cython.int, visibility="public")
    board = cython.declare(cython.int[:], visibility="public")
    board_hash = cython.declare(cython.longlong, visibility="public")
    positions = cython.declare(cython.tuple[cython.list, cython.list], visibility="public")
    last_action = cython.declare(cython.tuple, visibility="public")

    winner: cython.int

    """
    This class represents the game state for the Breakthrough board game.
    Breakthrough is a two-player game played on an 8x8 board.
    Each player starts with 16 pieces occupying the first two rows.
    The goal is to move one of your pieces to the opponent's home row.
    """

    def __init__(
        self,
        board=None,
        player=1,
        board_hash=0,
        positions=None,
        winner=0,
        last_action=(0, 0),
    ):
        """
        Initialize the game state with the given board configuration.
        If no board is provided, the game starts with the default setup.

        :param board: An optional board configuration.
        :param player: The player whose turn it is (1 or 2).
        """
        if board is None:
            self.board = self._init_board()
            # Keep track of the pieces on the board, this keeps evaluation and move generation from recomputing these over and over
            self.positions = where_is_k(self.board, 1), where_is_k(self.board, 2)
            self.last_action = (-1, -1)
            self.player = 1
            self.winner = 0
            self.board_hash = self._init_hash()
        else:
            self.board = board
            self.positions = positions
            self.player = player
            self.board_hash = board_hash
            self.winner = winner
            self.last_action = last_action

    def _init_board(self):
        board = np.zeros(64, dtype=np.int32)
        board[:16] = 2
        board[48:] = 1
        return board

    def _init_hash(self):
        board_hash = 0
        for position in range(64):
            player = self.board[position]
            row, col = divmod(position, 8)
            board_hash ^= BreakthroughGameState.zobrist_table[row][col][player]
        return board_hash

    @cython.cfunc
    def apply_action_playout(self, action: cython.tuple) -> cython.void:
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        move_t: cython.tuple[cython.int, cython.int] = action
        act_from: cython.int = move_t[0]
        act_to: cython.int = move_t[1]

        self.board[act_from] = 0  # Remove the piece from its current position
        captured_player: cython.int = self.board[act_to]  # Check for captures
        self.board[act_to] = self.player  # Place the piece at its new position

        # Keep track of the pieces on the board
        if captured_player != 0:
            self.positions[captured_player - 1].remove(act_to)
            # Keep track of the number of pieces of each player
            if len(self.positions[captured_player - 1]) == 0:
                self.winner = 3 - captured_player

        self.positions[self.player - 1].remove(act_from)
        self.positions[self.player - 1].append(act_to)

        if self.winner == 0:
            if self.player == 1 and act_to < 8:
                self.winner = 1
            elif self.player == 2 and act_to >= 56:
                self.winner = 2

        self.player = 3 - self.player
        self.last_action = action

    @cython.ccall
    def apply_action(self, action: cython.tuple) -> GameState:
        """
        Apply the given action to create a new game state. The current state is not altered by this method.
        Actions are represented as a tuple (from_position, to_position).

        :param action: The action to apply.
        :return: A new game state with the action applied.
        """
        new_board: cython.int[:] = self.board.copy()
        move_t: cython.tuple[cython.int, cython.int] = action
        act_from: cython.int = move_t[0]
        act_to: cython.int = move_t[1]

        new_board[act_from] = 0  # Remove the piece from its current position
        captured_player: cython.int = new_board[act_to]
        new_board[act_to] = self.player  # Place the piece at its new position

        new_positions: cython.tuple[cython.list, cython.list] = (
            self.positions[0].copy(),
            self.positions[1].copy(),
        )

        winner: cython.int = 0
        # Keep track of the pieces on the board
        if captured_player != 0:
            # Remove the to-position from the positions list
            new_positions[captured_player - 1].remove(act_to)
            if len(new_positions[captured_player - 1]) == 0:
                winner = 3 - captured_player  # The captured player has lost all pieces...

        new_positions[self.player - 1].remove(act_from)
        new_positions[self.player - 1].append(act_to)

        to_row: cython.int = act_to // 8
        to_col: cython.int = act_to % 8

        board_hash = (
            self.board_hash
            ^ BreakthroughGameState.zobrist_table[act_from // 8][act_from % 8][self.player]
            ^ BreakthroughGameState.zobrist_table[to_row][to_col][captured_player]
            ^ BreakthroughGameState.zobrist_table[to_row][to_col][self.player]
        )

        if winner == 0:
            if self.player == 1 and act_to < 8:
                winner = 1
            elif self.player == 2 and act_to >= 56:
                winner = 2

        return BreakthroughGameState(
            board=new_board,
            player=3 - self.player,
            board_hash=board_hash,
            positions=new_positions,
            winner=winner,
            last_action=action,
        )

    @cython.ccall
    def skip_turn(self) -> GameState:
        """Used for the null-move heuristic in alpha-beta search

        Returns:
            BreakthroughGameState: A new gamestate in which the players are switched but no move performed
        """
        # Pass the same board hash since this is only used for null moves
        return BreakthroughGameState(
            self.board.copy(),
            3 - self.player,
            board_hash=self.board_hash,
            positions=(
                self.positions[0].copy(),
                self.positions[1].copy(),
            ),
            winner=self.winner,
            last_action=self.last_action,
        )

    @cython.cfunc
    def get_random_action(self) -> cython.tuple:
        """
        Generate a single legal action for the current player.

        :return: A tuple representing a legal action (from_position, to_position).
        """

        dr: cython.int = -1
        if self.player == 2:
            dr = 1

        n: cython.int = len(self.positions[self.player - 1])
        start: cython.int = c_random(0, n - 1)  # This allows us to start at a random piece

        i: cython.int

        safe_captures: vector[pair[cython.int, cython.int]]
        all_moves: vector[pair[cython.int, cython.int]]
        captures: vector[pair[cython.int, cython.int]]
        all_moves.reserve(n * 3)

        for i in range(n):
            index: cython.int = (start + i) % n
            position: cython.int = self.positions[self.player - 1][index]

            row: cython.int = position // 8
            col: cython.int = position % 8

            start_dc: cython.int = c_random(0, 2)  # This allows us to start at in a random direction
            k: cython.int

            for k in range(3):
                dc: cython.int = dirs[(start_dc + k) % 3]
                new_row: cython.int = row + dr
                new_col: cython.int = col + dc
                if not (
                    (0 <= new_row) & (new_row < 8) & (0 <= new_col) & (new_col < 8)
                ):  # if the new position is not in bounds, skip to the next direction
                    continue
                new_position: cython.int = new_row * 8 + new_col

                # Straight, no capture or diagonal capture / empty cell
                if (dc == 0 and self.board[new_position] == 0) or (self.board[new_position] != self.player):
                    # Decisive moves
                    if (self.player == 1 and new_position < 8) or (self.player == 2 and new_position >= 56):
                        return (position, new_position)

                    # Captures
                    if new_position == (3 - self.player):
                        # Anti-decisive moves (captures from the last row)
                        if (self.player == 1 and position < 56) or (self.player == 2 and position < 8):
                            return (position, new_position)

                        # Prioritize safe captures
                        if is_safe(position, self.player, self.board):
                            safe_captures.push_back(pair[cython.int, cython.int](position, new_position))
                        else:
                            # give captures higher chance of being selected
                            captures.push_back(pair[cython.int, cython.int](position, new_position))

                    # Safe captures are prioritized anyway so no need to add non-captures
                    if safe_captures == [] and captures == []:
                        all_moves.push_back(pair[cython.int, cython.int](position, new_position))

        # Always do a safe capture if you can
        if not safe_captures.empty():
            if safe_captures.size() > 1:
                return safe_captures[c_random(0, safe_captures.size() - 1)]
            return safe_captures[0]
        if not captures.empty():
            if captures.size() > 1:
                return captures[c_random(0, captures.size() - 1)]
            return captures[0]
        # All_moves includes captures, they'll be selected with a higher probability
        return all_moves[c_random(0, all_moves.size() - 1)]

    @cython.ccall
    def get_legal_actions(self) -> cython.list[cython.tuple]:
        """
        Get all legal actions for the current player.

        :return: A list of legal actions as tuples (from_position, to_position).
        In case of a terminal state, an empty list is returned.
        """
        n: cython.int = len(self.positions[self.player - 1])
        legal_actions: vector[pair[cython.int, cython.int]]
        legal_actions.reserve(n * 3)
        dr: cython.int = -1
        if self.player == 2:
            dr = 1

        i: cython.int
        for i in range(n):
            position: cython.int = self.positions[self.player - 1][i]
            row: cython.int = position // 8
            col: cython.int = position % 8
            dc: cython.int
            for dc in range(-1, 2):
                new_row: cython.int = row + dr
                new_col: cython.int = col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:  # Check if the new position is within the board
                    new_position: cython.int = new_row * 8 + new_col

                    if dc == 0:  # moving straight
                        if self.board[new_position] == 0:
                            legal_actions.push_back(pair[cython.int, cython.int](position, new_position))

                    else:  # diagonal capture / move
                        if self.board[new_position] != self.player:
                            legal_actions.push_back(pair[cython.int, cython.int](position, new_position))
        return legal_actions

    @cython.ccall
    def is_terminal(self) -> cython.bint:
        """
        Check if the current game state is terminal (i.e., a player has won).

        :return: True if the game state is terminal, False otherwise.
        """
        if self.winner != 0:
            return 1
        return 0

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_reward(self, player: cython.int) -> cython.int:
        if self.winner == player:
            return win

        elif self.winner == 3 - player:
            return loss

        return draw

    @cython.cfunc
    def get_result_tuple(self) -> cython.tuple:
        if self.winner == 1:
            return (1.0, 0.0)
        elif self.winner == 2:
            return (0.0, 1.0)

        return (0.5, 0.5)

    @cython.cfunc
    def is_capture(self, move: cython.tuple) -> cython.bint:
        """
        Check if the given move results in a capture.

        :param move: The move to check as a tuple (from_position, to_position).
        :return: True if the move results in a capture, False otherwise.
        """
        # Check if the destination cell contains an opponent's piece
        i: cython.int = move[1]
        return self.board[i] == 3 - self.player

    # These dictionaries are used by run_games to set the parameter values
    param_order: dict = {
        "m_lorenz": 0,
        "m_mobility": 1,
        "m_safe": 2,
        "m_endgame": 3,
        "m_cap": 4,
        "m_piece_diff": 5,
        "m_opp_disc": 6,
        "m_decisive": 7,
        "a": 8,
    }

    default_params = array.array("d", [20.0, 2.0, 15.0, 11.0, 10.0, 13.0, 1.1, 800.0, 2000.0])

    @cython.cfunc
    @cython.exceptval(-9999999, check=False)
    def evaluate(
        self,
        player: cython.int,
        params: cython.double[:],
        norm: cython.bint = 0,
    ) -> cython.double:
        """
        Evaluate the board
        """
        board_values: cython.double = 0
        mobility_values: cython.double = 0
        safety_values: cython.double = 0
        endgame_values: cython.double = 0
        piece_diff: cython.double = 0
        decisive_values: cython.double = 0
        caps: cython.double = 0
        opponent: cython.int = 3 - player

        k: cython.int
        i: cython.int

        for k in range(2):
            positions: cython.list = self.positions[k]

            for i in range(len(positions)):
                position: cython.int = positions[i]
                piece: cython.int = k + 1
                # Player or opponent
                if piece == player:
                    multiplier: cython.double = 1.0
                else:
                    multiplier: cython.double = -1.0

                if piece == 2:
                    piece_value: cython.double = lorentz_values[position]
                else:
                    piece_value: cython.double = lorentz_values[63 - position]

                board_values += multiplier * piece_value

                if params[1] != 0.0:
                    mob_val: cython.double = piece_mobility(position, piece, self.board)
                    mobility_values += multiplier * mob_val * (1 + piece_value)

                # TODO Je moet eigenlijk belonen voor niet safe
                if params[2] != 0.0 and not is_safe(position, piece, self.board):
                    safety_values -= multiplier * piece_value

        piece_diff = len(self.positions[player - 1]) - len(self.positions[opponent - 1])

        if params[3] != 0.0 and (len(self.positions[player - 1]) + len(self.positions[opponent - 1])) < 12:
            endgame_values = params[3] * piece_diff
        # "m_lorenz": 0,
        #         "m_mobility": 1,
        #         "m_safe": 2,
        #         "m_endgame": 3,
        #         "m_cap": 4,
        #         "m_piece_diff": 5,
        #         "m_opp_disc": 6,
        #         "m_decisive": 7,
        #         "a": 8,
        if params[7] != 0.0:
            player_1_decisive, player_2_decisive = is_decisive(self)

            if player == 1:
                decisive_values = params[7] if player_1_decisive else 0
                decisive_values -= params[7] if player_2_decisive else 0
            else:  # self.player == 2
                decisive_values = params[7] if player_2_decisive else 0
                decisive_values -= params[7] if player_1_decisive else 0

        if params[4] >= 0:
            caps = count_capture_moves(self.board, self.positions[player - 1], player) - count_capture_moves(
                self.board, self.positions[opponent - 1], opponent
            )

        # 'm_lorenz': 0,
        # 'm_mobility': 1,
        # 'm_safe': 2,
        # 'm_endgame': 3,
        # 'm_cap': 4,
        # 'm_piece_diff': 5,
        # 'm_opp_disc': 6,
        # 'm_decisive': 7,
        # 'a': 8

        eval_value: cython.double = (
            decisive_values
            + params[0] * board_values
            + params[1] * mobility_values
            + params[2] * safety_values
            + endgame_values
            + params[4] * caps
            + params[5] * piece_diff
        )
        # print the individual values for debugging
        print(
            f"{player=} decisive: {decisive_values:.3f} | board: {board_values:.3f} | mobility: {mobility_values:.3f} | safety: {safety_values:.3f} | endgame: {endgame_values:.3f} | caps: {caps:.3f} | piece_diff: {piece_diff:.3f}"
        )

        if self.player == opponent:
            eval_value *= params[6]

        if norm:
            return normalize(eval_value, params[8])
        else:
            return eval_value

    @cython.cfunc
    def evaluate_moves(self, moves: cython.list) -> cython.list:
        """
        Evaluate the given moves using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param moves: The list of moves to evaluate.
        :return: The list of heuristic values of the moves.
        """
        scores: cython.list = [()] * len(moves)
        i: cython.int
        for i in range(len(moves)):
            scores[i] = (moves[i], self.evaluate_move(moves[i]))
        return scores

    @cython.cfunc
    def move_weights(self, moves: cython.list) -> cython.list:
        n_moves = len(moves)

        scores: vector[cython.int]
        scores.reserve(n_moves)
        i: cython.int
        for i in range(n_moves):
            move: cython.tuple = moves[i]
            scores.push_back(self.evaluate_move(move))
        return scores

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def evaluate_move(self, move: cython.tuple) -> cython.int:
        """
        Evaluate the given move using a simple heuristic: each step forward is worth 1 point,
        and capturing an opponent's piece is worth 2 points.

        :param move: The move to evaluate.
        :return: The heuristic value of the move.
        """
        score: cython.int = 0
        from_position: cython.int = move[0]
        to_position: cython.int = move[1]
        # Use lorentz_values for base_value, the max value is 36, so we add 36 to the score to make sure it is non-negative
        if self.player == 2:
            score = 36 + (lorentz_values[to_position] - lorentz_values[from_position])
        else:
            # Player 1 views the lorenz_values in reverse
            score = 36 + (lorentz_values[63 - to_position] - lorentz_values[63 - from_position])

        # Reward capturing
        if self.board[to_position] == 3 - self.player:
            score = int(score**2)  # square score if it's a capture
            # An antidecisive move
            if (self.player == 1 and from_position >= 56) or (self.player == 2 and from_position < 8):
                score += 1000000  # Add a very high score if the move is antidecisive

        # Reward safe positions
        if is_safe(to_position, self.player, self.board):
            score *= 2  # Add base_value again if the position is safe

        # Reward decisive moves
        # The decisive condition checks for a piece on the penultimate row that can move to the opponent's final row without the opportunity of being captured.
        if (self.player == 1 and (8 <= from_position <= 16)) or (
            self.player == 2 and (48 <= from_position <= 56)
        ):
            score += 1000000  # Add a very high score if the move is decisive

        return score

    @cython.ccall
    def visualize(self, full_debug=False):
        """
        Visualize the board for the Breakthrough game.
        """
        output = ""
        cell_representation = {0: ".", 1: "W", 2: "B"}
        column_letters = "  " + " ".join("ABCDEFGH") + "\n"  # Use chess-style notation

        for i in range(8):
            row = [cell_representation.get(piece, ".") for piece in self.board[i * 8 : i * 8 + 8]]
            formatted_row = " ".join(row)
            output += f"{i + 1} {formatted_row}\n"

        output += column_letters
        output += f"Player: {self.player}\n"

        if full_debug:
            if not self.validate_actions():
                print(" !!! Invalid actions detected !!! ")
            output += "..." * 10 + "debug info" + "..." * 10 + "\n"
            output += "Hash: " + str(self.board_hash) + " | "
            output += f"reward: {self.get_reward(1)}/{self.get_reward(2)} | terminal: {self.is_terminal()}\n"

            moves = self.get_legal_actions()
            output += f"{len(moves)} actions | "
            # ---------------------------Evaluations---------------------------------
            output += f"evaluation: {self.evaluate(1, norm=False, params=self.default_params):.3f}/{self.evaluate(2, norm=False, params=self.default_params):.3f}"
            output += f" | normalized: {self.evaluate(1, norm=True, params=self.default_params):.4f}/{self.evaluate(2, norm=True, params=self.default_params):.4f}\n"
            output += "..." * 20 + "\n"

            white_pieces = self.positions[0]
            black_pieces = self.positions[1]
            output += f"# white pieces: {len(white_pieces)} | # black pieces: {len(black_pieces)} | "
            # ---------------------------Endgame-------------------------------------------
            output += f"is endgame: {np.count_nonzero(self.board) < 16} | "
            # ---------------------------Possible captures---------------------------------
            wh_caps = count_capture_moves(self.board, white_pieces, 1)
            bl_caps = count_capture_moves(self.board, black_pieces, 2)
            output += f"White has {wh_caps} capture value, black has {bl_caps} \n"
            # ---------------------------Decisiveness---------------------------------
            p1_decisive, p2_decisive = is_decisive(self)
            output += f"Player 1 decisive: {p1_decisive}, Player 2 decisive: {p2_decisive}\n"
            # ---------------------------Piece safety--------------------------------------
            output += "Unsafe pieces:\n"
            for piece in black_pieces:
                if not is_safe(piece, 2, self.board):
                    output += f"{self.readable_location(piece)} \n"
            for piece in white_pieces:
                if not is_safe(piece, 1, self.board):
                    output += f"{self.readable_location(piece)} \n"

            # ---------------------------Piece mobility---------------------------------
            output += "Blocked pieces:\n black: "
            for piece in black_pieces:
                mob = piece_mobility(piece, 2, self.board)
                if mob == 0:
                    output += f"{self.readable_location(piece)},"
            output += " | white: "
            for piece in white_pieces:
                mob = piece_mobility(piece, 1, self.board)
                if mob == 0:
                    output += f"{self.readable_location(piece)},"
            output += "\n" + "..." * 20 + "\n"

        return output

    @cython.ccall
    @cython.infer_types(True)
    def validate_actions(self):
        """
        Check if the available actions are valid or not.

        :return: A boolean indicating whether all actions are valid or not.
        """
        actions = self.get_legal_actions()
        # Make sure all actions are unique.
        if len(actions) != len(set(actions)):
            print("Found duplicate actions.")
            return False

        for action in actions:
            from_position: cython.int = action[0]
            to_position: cython.int = action[1]

            # Check that there is a piece belonging to the current player at the 'from' position.
            if self.board[from_position] != self.player:
                print(f"No piece for current player at {to_chess_notation(from_position)}.")
                return False

            # Check that there is not a piece belonging to the current player at the 'to' position.
            if self.board[to_position] == self.player:
                print(f"Existing piece for current player at {to_chess_notation(to_position)}.")
                return False

            # Check that all moves are in the correct direction (higher rows for p2, lower rows for p1).
            if self.player == 1 and to_position >= from_position:
                print(f"Invalid move direction for player 1 at {self.readable_move(action)}.")
                return False
            elif self.player == 2 and to_position <= from_position:
                print(f"Invalid move direction for player 2 at {self.readable_move(action)}.")
                return False

        return True

    def readable_move(self, move):
        """
        returns a move in a more human-friendly format.

        :param move: A tuple (from_position, to_position) in board-index style.
        """

        from_position, to_position = move
        return f"{to_chess_notation(from_position)} -> {to_chess_notation(to_position)}"

    def readable_location(self, position):
        return to_chess_notation(position)

    @property
    def transposition_table_size(self):
        # return an appropriate size based on the game characteristics
        return 2**24

    def __repr__(self) -> str:
        return "breakthrough"


def to_chess_notation(index):
    """Transform a board index into chess notation."""
    row, col = divmod(index, 8)
    return f"{chr(col + 65)}{row + 1}"


@cython.cfunc
@cython.locals(
    player_1_decisive=cython.int,
    player_2_decisive=cython.int,
    pos=cython.int,
    i=cython.int,
)
@cython.returns(tuple[cython.int, cython.int])
def is_decisive(state: BreakthroughGameState):
    player_1_decisive = player_2_decisive = 0

    for i in range(len(state.positions[0])):
        pos = state.positions[0][i]
        # Decisive moves
        if 8 <= pos < 16:
            player_1_decisive += 1
            if state.player == 1:  # Sure win
                player_1_decisive += 10

    for i in range(len(state.positions[1])):
        pos = state.positions[1][i]
        # Decisive moves
        if 48 < pos <= 56:
            player_2_decisive += 1
            if state.player == 2:  # Sure win
                player_2_decisive += 10

    return player_1_decisive, player_2_decisive


BL_DIR = [[1, 0], [1, -1], [1, 1]]
WH_DIR = [[-1, 0], [-1, -1], [-1, 1]]


@cython.cfunc
@cython.exceptval(-1, check=False)
@cython.locals(
    position=cython.int,
    player=cython.int,
    board=cython.int[:],
    row=cython.int,
    col=cython.int,
    mobility=cython.int,
    directions=cython.int[3][3],
    dr=cython.int,
    dc=cython.int,
    new_row=cython.int,
    new_col=cython.int,
    new_position=cython.int,
)
def piece_mobility(position, player, board) -> cython.int:
    """
    Calculates the mobility of a piece at the given position.

    :param position: The position of the piece on the board.
    :param player: The player to which the piece belongs (1 or 2).
    :param board: The game state.
    :return: The mobility value of the piece.
    """
    row = position // 8
    col = position % 8
    mobility = 0

    # The directions of the moves depend on the player.
    # White (1) moves up (to lower rows), Black (2) moves down (to higher rows).
    # if player == 1:  # White
    #     directions = WH_DIR  # Forward, and diagonally left and right
    # else:  # Black
    #     directions = BL_DIR  # Forward, and diagonally left and right

    dr = -1
    if player == 2:
        dr = 1

    for i in range(-1, 2):
        dc = i
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < 8 and 0 <= new_col < 8:
            new_position = new_row * 8 + new_col
            if (dc == 0 and board[new_position] == 0) or (abs(dc) == 1 and board[new_position] != player):
                mobility += 1
    return mobility


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
@cython.locals(
    position=cython.int,
    player=cython.int,
    board=cython.int[:],
    x=cython.int,
    y=cython.int,
    row_direction=cython.int,
    col_direction=cython.int,
    attackers=cython.int,
    defenders=cython.int,
    opponent=cython.int,
    i=cython.int,
    new_x=cython.int,
    new_y=cython.int,
    new_position=cython.int,
    piece=cython.int,
)
def is_safe(position, player, board) -> cython.bint:
    """
    Determines if a piece at a given position is safe from being captured.

    :param position: The position of the piece.
    :param player: The player to which the piece belongs (1 or 2).
    :param board: The game board.
    :return: True if the piece is safe, False otherwise.
    """

    # Convert linear position to 2D grid coordinates.
    x = position // 8
    y = position % 8

    # Initialize counters for opponent pieces (attackers) and own pieces (defenders) around the current piece.
    attackers = 0
    defenders = 0

    # Determine opponent's player number
    opponent = 3 - player

    # Define directions for row and column
    row_direction = -1
    if player == 2:
        row_direction = 1
    col_direction = -1  # start with -1 direction, then change to 1 in the loop

    # For each direction (-1, 1), check the positions in the row direction (straight and diagonals).
    for _ in range(2):
        # Check the first row direction (upwards for player 1, downwards for player 2).
        new_x = x + row_direction
        new_y = y + col_direction

        # If the new position is within the board boundaries
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            # Calculate linear position from grid coordinates.
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            # If there's an opponent piece in this position, increment the attackers counter.
            if piece == opponent:
                attackers += 1

        # Check the second row direction (downwards for player 1, upwards for player 2).
        new_x = x - row_direction
        new_y = y + col_direction

        # If the new position is within the board boundaries
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            # Calculate linear position from grid coordinates.
            new_position = new_x * 8 + new_y
            piece = board[new_position]

            # If there's a friendly piece in this position, increment the defenders counter.
            if piece == player:
                defenders += 1

        # Change the column direction for the second iteration
        col_direction = 1

    # if attackers <= defenders:
    #     print(f"{to_chess_notation(position)}: {attackers=} <= {defenders=}")
    # The piece is considered safe if the number of attackers is less than or equal to the number of defenders.
    return attackers <= defenders


@cython.cfunc
@cython.exceptval(-1, check=False)
@cython.locals(
    board=cython.int[:],
    positions=cython.list,
    player=cython.int,
    total_capture_moves=cython.int,
    position=cython.int,
    row=cython.int,
    col=cython.int,
    dr=cython.int,
    piece_capture_moves=cython.int,
    dc=cython.int,
    new_row=cython.int,
    new_col=cython.int,
    new_position=cython.int,
    i=cython.int,
    n=cython.int,
    opponent=cython.int,
    piece_value=cython.int,
)
def count_capture_moves(board, positions, player) -> cython.int:
    """Count the number of pieces that can capture and the total number of possible capture moves for a given player.

    :param game_state: The game state instance.
    :param player: The player (1 or 2) to count capture moves for.
    :return: The number of pieces that can capture and the total number of possible capture moves for the player.
    """
    total_capture_moves = 0
    opponent = 3 - player
    n: cython.int = len(positions)

    for i in range(n):
        position = positions[i]
        row = position // 8
        col = position % 8

        dr = -1
        if player == 2:
            dr = 1

        piece_capture_moves = 0

        for dc in range(-1, 2, 2):  # Only diagonal movements can result in capture
            new_row = row + dr
            new_col = col + dc

            if (
                0 <= new_row < 8 and 0 <= new_col < 8
            ):  # Check if the new position is within the board boundaries
                new_position = new_row * 8 + new_col
                # Check if the new position contains an opponent's piece
                if board[new_position] == opponent:
                    # Add the lorenz value from the opponent's perspective
                    # Assume that we'll capture the most valuable piece
                    piece_value = (
                        lorentz_values[new_position] if opponent == 2 else lorentz_values[63 - new_position]
                    )
                    piece_capture_moves = max(piece_capture_moves, piece_value)

        total_capture_moves += piece_capture_moves

    return int(total_capture_moves)


@cython.ccall
def evaluate_breakthrough(
    state: BreakthroughGameState,
    player: cython.int,
    m_piece: cython.double = 1.0,
    m_dist: cython.double = 1.0,
    m_near: cython.double = 1.0,
    m_blocked: cython.double = 0.25,
    m_opp_disc: cython.double = 0.9,
    a: cython.int = 20,
    norm: cython.bint = 0,
) -> cython.double:
    """
    Evaluates the current game state for Breakthrough using a custom evaluation function,
    which considers number of pieces, average distance of pieces to the opponent's side,
    number of pieces close to the opponent's side, and number of blocked pieces.

    :param state: The game state to evaluate.
    :param player: The player to evaluate for (1 or 2).
    :param m_piece: Weight assigned to the difference in number of pieces.
    :param m_dist: Weight assigned to the difference in average distance to opponent's side.
    :param m_near: Weight assigned to the difference in number of pieces close to opponent's side.
    :param m_blocked: Weight assigned to the difference in number of blocked pieces.
    :param m_opp_disc: Multiplier for the evaluation score if it's the opponent's turn.
    :param a: Normalization factor for the evaluation score.
    :param norm: If True, the function will return a normalized evaluation score.

    :return: The evaluation score for the given player.
    """

    player_distance: cython.int = 0
    player_near_opponent_side: cython.int = 7
    player_blocked: cython.int = 0
    opponent_distance: cython.int = 0
    opponent_near_opponent_side: cython.int = 7
    opponent_blocked: cython.int = 0
    board: cython.int[:] = state.board
    new_position: cython.int
    new_col: cython.int
    new_row: cython.int
    y: cython.int
    x: cython.int
    piece: cython.int
    dr: cython.int
    dc: cython.int
    dist: cython.int
    k: cython.int

    for k in range(2):
        positions: cython.list = state.positions[k]
        for i in range(len(positions)):
            pos: cython.int = positions[i]
            piece = board[pos]
            x = positions[i] // 8
            y = positions[i] % 8

            if piece == player:
                dist = 7 - x if piece == 2 else x
                player_distance += dist

                if (piece == 2 and x >= 4) or (piece == 1 and x <= 3):
                    player_near_opponent_side = min(dist, player_near_opponent_side)

            else:
                dist = 7 - x if piece == 2 else x
                opponent_distance += dist

                if (piece == 2 and x >= 4) or (piece == 1 and x <= 3):
                    opponent_near_opponent_side = min(dist, opponent_near_opponent_side)

            # Determine the direction of movement based on the current player
            if piece == 1:
                dr = -1
            else:
                dr = 1

            blocked = True
            for dc in range(-1, 2):
                new_row = x + dr
                new_col = y + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    new_position = new_row * 8 + new_col
                    if board[new_position] == 0 or (dc != 0 and board[new_position] != piece):
                        blocked = False
                        break

            if piece == player:
                player_blocked += 1 if blocked else 0
            else:
                opponent_blocked += 1 if blocked else 0

    eval_value = m_piece * (len(state.positions[player - 1]) - len(state.positions[(3 - player) - 1]))
    eval_value += m_dist * (opponent_distance - player_distance)
    eval_value += m_near * (player_near_opponent_side - opponent_near_opponent_side)
    eval_value += m_blocked * (opponent_blocked - player_blocked)

    if state.player != player:
        eval_value *= m_opp_disc

    if norm:
        return normalize(eval_value, a)
    else:
        return eval_value
