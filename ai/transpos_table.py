from collections import OrderedDict
from typing import Tuple, Union, Optional
import weakref

import numpy as np


class IncorrectBoardException(Exception):
    pass


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
        self, key: int, depth: int, player: str, max_d: int, board: Optional[str] = None
    ) -> Tuple[float, tuple]:
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
        try:
            value, stored_depth, stored_player, best_move, stored_max_d, stored_board = self.table[key]
            if stored_board is not None and not np.array_equal(stored_board, board):
                raise IncorrectBoardException(
                    f"Stored: {stored_board} is not the same as {board} stored at {key}"
                )

            if stored_depth >= depth and stored_player == player and stored_max_d >= max_d:
                # Add the key to the LRU again so it it "refreshed"
                self.table[key] = (
                    value,
                    stored_depth,
                    stored_player,
                    best_move,
                    stored_max_d,
                    stored_board,
                )
                self.cache_hits += 1  # Increase the cache hits
                return value, best_move

        except KeyError:
            self.cache_misses += 1  # Increase the cache misses
        return None, None

    def put(
        self,
        key: int,
        value: float,
        depth: int,
        player: str,
        best_move: Tuple[int, int],
        max_d: int,
        board: Optional[str] = None,
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
        try:
            (
                _,
                stored_depth,
                _,
                _,
                stored_max_d,
                _,
            ) = self.table[key]
            if stored_depth > depth and stored_max_d >= max_d:
                return  # The stored value is from a deeper depth, don't overwrite it.
            else:
                self.table.pop(key)
                self.collisions += 1  # Increase the collisions

        except KeyError:
            if len(self.table) == self.size:
                self.table.popitem(last=False)
                self.cleanups += 1

        self.table[key] = (value, depth, player, best_move, max_d, board)

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


from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np


class IncorrectBoardException(Exception):
    pass


class TranspositionTableMCTS:
    def __init__(self, size):
        """
        Initialize the transposition table with the given size.

        :param size: The maximum number of entries the transposition table can hold.
        """
        self.size = size
        self.table = OrderedDict()
        self.visited = []  # Hashes of the visited states
        # Metrics for debugging and experimental purposes
        self.c_cache_hits = self.c_cache_misses = self.c_collisions = self.c_cleanups = 0
        self.cache_hits = self.cache_misses = self.collisions = self.cleanups = 0

    def exists(self, key):
        return key in self.table

    def get(self, key: int, board=None):
        try:
            entries = self.table[key]
            for entry in entries:
                v1, v2, im_value, visits, solved_player, is_expanded, _board_ref = entry
                _board = _board_ref()  # Resolve weak reference

                if _board is not None and not np.array_equal(_board, board):
                    continue  # Skip this entry, as it is a collision

                # Refresh entry in LRU
                entries.remove(entry)
                entries.append(entry)
                self.visited.append(key)  # Add this hash to the visited set
                self.cache_hits += 1
                return v1, v2, im_value, visits, solved_player, is_expanded

        except KeyError:
            self.cache_misses += 1

        return None, None, None, None, None, None

    def put(
        self,
        key: int,
        v1: float = 0,
        v2: float = 0,
        visits: int = 0,
        solved_player: int = None,
        is_expanded: bool = False,
        im_value: float = None,
        board=None,
    ):
        entries = self.table.get(key, [])
        board_ref = weakref.ref(board)  # Create weak reference
        self.visited.append(key)  # Add this hash to the visited set

        for entry in entries:
            _v1, _v2, _im_value, _visits, _solved_player, _is_expanded, _board_ref = entry
            _board = _board_ref()  # Resolve weak reference

            if _board is not None and not np.array_equal(_board, board):
                self.collisions += 1
                continue  # Skip this entry, as it is a collision

            # Update existing entry
            entry[:] = (
                v1 + _v1,
                v2 + _v2,
                im_value or _im_value,
                visits + _visits,
                solved_player or _solved_player,
                is_expanded or _is_expanded,
                board_ref,
            )
            return

        # Create new entry if no existing entry was found
        entries.append([v1, v2, im_value, visits, solved_player, is_expanded, board_ref])
        self.table[key] = entries

    def evict(self):
        # Replace the table with a new table that only includes keys in the visited set
        self.table = {key: self.table[key] for key in set(self.visited)}
        self.num_entries = sum(len(entries) for entries in self.table.values())  # Update num_entries
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
