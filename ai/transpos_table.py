# cython: language_level=3
# cython: boundscheck=False

import collections

from collections import OrderedDict
from typing import DefaultDict

import cython

if cython.compiled:
    print("Transpositions is compiled.")
else:
    print("Transpositions is just a lowly interpreted script.")


class IncorrectBoardException(Exception):
    pass


@cython.cclass
class MoveHistory:
    """
    Class to manage a move history table for a game. The table records how many times
    each move has led to alpha cutoffs in an alpha-beta search.

    The table is implemented as a default dictionary, which allows for efficient updates
    and retrieval of move histories.
    """

    table: DefaultDict  # The table itself

    def __init__(self):
        """
        Initialize the move history table.
        """
        self.table = collections.defaultdict(int)

    def get(self, move):
        """
        Retrieve the history of a move from the move history table.

        :param move: The move whose history is to be retrieved.
        :return: The history of the move, or 0 if the move has no history.
        """
        return self.table[move]

    def update(self, move, increment):
        """
        Update the history of a move in the move history table.

        :param move: The move whose history is to be updated.
        :param increment: The amount by which the move's history is to be increased.
        """
        self.table[move] += increment


class TranspositionTable:
    """
    Class to manage a transposition table for a game. The table stores previously computed values
    of the game's state evaluation function for a given board state, player, and search depth.

    The table is implemented as an ordered dictionary, which allows it to maintain the insertion order
    and efficiently remove the oldest entry when necessary.

    Collisions in the table are handled by storing and comparing board states when provided.
    """

    def __init__(self, size):
        """
        Initialize the transposition table with the given size.

        :param size: The maximum number of entries the transposition table can hold.
        """
        self.size = size
        self.table = OrderedDict()
        # Metrics for debugging and experimental purposes
        self.c_cache_hits = self.c_cache_misses = self.c_collisions = self.c_cleanups = 0
        self.cache_hits = self.cache_misses = self.collisions = self.cleanups = 0

    def get(
        self,
        key,
        depth,
        player,
        board=None,
    ) -> cython.tuple:
        """
        Retrieve a value from the transposition table for the given key, depth, and player.
        If a board is provided, it is also compared with the stored board for the same key.

        :param key: The hash of the board state.
        :param depth: The depth of the search when the value was stored.
        :param player: The player for whom the value was computed.
        :param board: The board state, used for collision detection (default: None).
        :return: The stored value and best_move, or None if there is no value for the given key or the stored depth is less than the provided depth.
        :raise IncorrectBoardException: If a board is provided and does not match the stored board for the same key.

        Example:
            key = 'some_key'
            depth = 3
            player = 'P1'
            board = 'board_state'
            transposition_table.get(key, depth, player, board)
        """
        dict_val: cython.tuple
        try:
            dict_val = self.table[key]
            value: cython.float = dict_val[0]
            stored_depth: cython.uint = dict_val[1]
            stored_player: cython.int = dict_val[2]
            best_move: cython.tuple = dict_val[3]
            stored_board: cython.str = dict_val[4]

            # value, stored_depth, stored_player, best_move, stored_max_d, stored_board = self.table[key]
            # if stored_board is not None and not np.array_equal(stored_board, board):
            #     raise IncorrectBoardException(
            #         f"Stored: {stored_board} is not the same as {board} stored at {key}"
            #     )

            if stored_depth >= depth and stored_player == player:
                # Add the key to the LRU again so it it "refreshed"
                self.table[key] = (
                    value,
                    stored_depth,
                    stored_player,
                    best_move,
                    stored_board,
                )
                self.cache_hits += 1  # Increase the cache hits
                return value, best_move

        except KeyError:
            self.cache_misses += 1  # Increase the cache misses
        return 0.0, ()

    def put(
        self,
        key,
        value,
        depth,
        player,
        best_move,
        board=None,
    ):
        """
        Insert a value into the transposition table for the given key, depth, and player.
        If the table is full, the oldest entry is removed.
        If a board is provided, it is also stored for collision detection.

        :param key: The hash of the board state.
        :param value: The value to be stored.
        :param depth: The depth of the search when the value was computed.
        :param player: The player for whom the value was computed.
        :param best_move: The best move for the player in this state.
        :param board: The board state, used for collision detection (default: None).

        Example:
            key = 'some_key'
            value = 10
            depth = 3
            player = 'P1'
            best_move = (2, 3)
            board = 'board_state'
            transposition_table.put(key, value, depth, player, best_move, board)
        """
        stored_depth: cython.uint
        try:
            (
                _,
                stored_depth,
                _,
                _,
                _,
            ) = self.table[key]

            self.collisions += 1  # Increase the collisions

            if stored_depth > depth:
                return  # The stored value is from a deeper depth, don't remove it.
            else:
                self.table.pop(key)

        except KeyError:
            if len(self.table) == self.size:
                self.table.popitem(last=False)
                self.cleanups += 1

        self.table[key] = (value, depth, player, best_move, board)

    def reset_metrics(self):
        # Keep some cumulative statistics
        self.c_cache_hits += self.cache_hits
        self.c_cache_misses += self.cache_misses
        self.c_collisions += self.collisions
        self.c_cleanups += self.cleanups
        # Metrics for debugging and experimental purposes
        self.cache_hits = 0
        self.cache_misses = 0
        self.collisions = 0
        self.cleanups = 0

    def get_metrics(self):
        """
        Get the metrics for debugging and experimental purposes.
        """
        return {
            "tt_cache_hits": self.cache_hits,
            "tt_cache_misses": self.cache_misses,
            "tt_entries": len(self.table),
            "tt_size": self.size,
            "tt_collisions": self.collisions,
            "tt_cleanups": self.cleanups,
        }

    def get_cumulative_metrics(self):
        return {
            "tt_total_cache_hits": self.c_cache_hits,
            "tt_total_cache_misses": self.c_cache_misses,
            "tt_total_collisions": self.c_collisions,
            "tt_total_cleanups": self.c_cleanups,
            "tt_size": self.size,
        }


import numpy as np


class TranspositionTableMCTS:
    def __init__(self, size):
        """
        Initialize the transposition table with the given size.

        :param size: The maximum number of entries the transposition table can hold.
        """

        self.table = np.zeros(shape=(size, 6), dtype=np.float64)
        self.size = size
        self.puts = 0
        self.gets = 0
        self.uniques = 0

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    def get(self, key):
        self.gets += 1
        return self.table[key % self.size]

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    def put(self, key, v1=0, v2=0, visits=0, solved_player=0, is_expanded=0, eval_value=0):
        entry: cython.double[:] = self.table[key % self.size]
        self.puts += 1

        if not any(entry):  # If the entry is empty, we have a new entry
            # Create new entry if no existing entry was found
            entry[0] = v1
            entry[1] = v2
            entry[2] = visits
            entry[3] = solved_player
            entry[4] = is_expanded
            entry[5] = eval_value
            self.uniques += 1
            return

        if solved_player != 0.0:
            assert (
                entry[3] == 0 or entry[3] == solved_player
            ), f"Trying to overwrite a previously solved position.. was: {entry[3]}, would become: {solved_player}"

            entry[3] = solved_player

        # Update existing entry
        entry[0] += v1
        entry[1] += v2
        entry[2] += visits
        if entry[4] == 0.0:  # If the entry is already expanded, don't overwrite it
            entry[4] = is_expanded
        if eval_value != 0.0:  # Overwrite only if a value is passed
            entry[5] = eval_value

    def reset_metrics(self):
        self.puts = 0
        self.gets = 0
        self.uniques = 0

    def get_metrics(self):
        return {
            "tt_gets": self.gets,
            "tt_puts": self.puts,
            "tt_uniques": self.uniques,
            "tt_size": self.size,
        }

    def get_cumulative_metrics(self):
        pass


class TranspositionTableMCTSDict:
    def __init__(self, size):
        """
        Initialize the transposition table with the given size.

        :param size: The maximum number of entries the transposition table can hold.
        """
        # TODO Hier kun je beter een memoryview van maken
        self.table = dict()
        self.visited = set()  # Hashes of the visited states
        # Metrics for debugging and experimental purposes
        self.c_cache_hits = self.c_cache_misses = self.c_collisions = self.c_cleanups = 0
        self.cache_hits = self.cache_misses = self.collisions = self.cleanups = 0

    def exists(self, key):
        return key in self.table

    def get(self, key):
        try:
            entry: tuple[
                cython.double,
                cython.double,
                cython.double,
                cython.int,
                cython.bint,
                cython.double,
            ] = self.table[key]

            self.visited.add(key)  # Add this hash to the visited set
            self.cache_hits += 1

            return entry

        except KeyError:
            self.cache_misses += 1
            return (0.0, 0.0, 0.0, 0, 0, 0.0)

    @cython.locals(
        _v1=cython.double,
        _v2=cython.double,
        _visits=cython.double,
        _solved_player=cython.int,
        _is_expanded=cython.bint,
        _eval_value=cython.double,
    )
    def put(self, key, v1=0, v2=0, visits=0, solved_player=0, is_expanded=0, eval_value=0):
        try:
            entry: tuple[
                cython.double,
                cython.double,
                cython.double,
                cython.int,
                cython.bint,
                cython.double,
            ] = self.table[key]

            _v1, _v2, _visits, _solved_player, _is_expanded, _eval_value = entry

            if eval_value != 0:  # Overwrite if a new values is passed
                _eval_value = eval_value

            if solved_player != 0:
                assert (
                    _solved_player == 0 or _solved_player == solved_player
                ), f"Trying to overwrite a previously solved position.. was: {_solved_player}, would become: {solved_player}"
                _solved_player = solved_player

            # Update existing entry
            self.table[key] = (
                v1 + _v1,
                v2 + _v2,
                visits + _visits,
                _solved_player,
                is_expanded or _is_expanded,
                _eval_value,
            )
        except KeyError:
            self.num_entries += 1
            # Create new entry if no existing entry was found
            self.table[key] = (v1, v2, visits, solved_player, is_expanded, eval_value)
        finally:
            self.visited.add(key)  # Add this hash to the visited set

    def evict(self):
        # Replace the table with a new table that only includes keys in the visited set
        entries_before = len(self.table)
        self.table = {key: self.table[key] for key in set(self.visited)}
        self.num_entries = len(self.table)  # Update num_entries
        self.evicted = entries_before - self.num_entries  # Update evicted
        self.visited.clear()  # Clear the visited set

    def reset_metrics(self):
        # Keep some cumulative statistics
        self.c_cache_hits += self.cache_hits
        self.c_cache_misses += self.cache_misses
        self.c_collisions += self.collisions
        self.c_cleanups += self.cleanups
        # Metrics for debugging and experimental purposes
        self.cache_hits = 0
        self.cache_misses = 0
        self.collisions = 0
        self.cleanups = 0

    def get_metrics(self):
        """
        Get the metrics for debugging and experimental purposes.
        """
        return {
            "tt_cache_hits": self.cache_hits,
            "tt_cache_misses": self.cache_misses,
            "tt_entries": len(self.table),
            "tt_visited": len(self.visited),
            "tt_collisions": self.collisions,
            "tt_cleanups": self.cleanups,
            "tt_evicted": self.evicted,
        }

    def get_cumulative_metrics(self):
        pass
