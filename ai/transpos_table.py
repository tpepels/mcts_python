from collections import OrderedDict


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
        self.reset_metrics()

    def get(self, key, depth, player, board=None):
        """
        Retrieve a value from the transposition table for the given key, depth, and player.
        If a board is provided, it is also compared with the stored board for the same key.

        :param key: The hash of the board state.
        :param depth: The depth of the search when the value was stored.
        :param player: The player for whom the value was computed.
        :param board: The board state, used for collision detection (default: None).
        :return: The stored value, or None if there is no value for the given key or the stored depth is less than the provided depth.
        :raise IncorrectBoardException: If a board is provided and does not match the stored board for the same key.
        """
        try:
            value, stored_depth, stored_player, stored_board = self.table.pop(key)
            if stored_board is not None and stored_board != board:
                raise IncorrectBoardException(
                    f"Stored: {stored_board} is not the same as {board} stored at {key}"
                )

            if stored_depth >= depth:
                if stored_player != player:
                    value = -value

                self.table[key] = (value, stored_depth, stored_player, stored_board)
                self.cache_hits += 1  # Increase the cache hits

                return value
        except KeyError:
            self.cache_misses += 1  # Increase the cache misses
            return None

    def put(self, key, value, depth, player, board=None):
        """
        Insert a value into the transposition table for the given key, depth, and player.
        If the table is full, the oldest entry is removed.
        If a board is provided, it is also stored for collision detection.

        :param key: The hash of the board state.
        :param value: The value to be stored.
        :param depth: The depth of the search when the value was computed.
        :param player: The player for whom the value was computed.
        :param board: The board state, used for collision detection (default: None).
        """
        if key in self.table:
            v = self.table[key]
            if v[1] >= depth:
                return  # The stored value is from a deeper depth, don't overwrite it.
            else:
                self.table.pop(key)
                self.collisions += 1  # Increase the collisions
        elif len(self.table) == self.size:
            self.table.popitem(last=False)
            self.cleanups += 1

        self.table[key] = (value, depth, player, board)

    def reset_metrics(self):
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
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "table_size": len(self.table),
            "collisions": self.collisions,
            "cleanups": self.cleanups,
        }
