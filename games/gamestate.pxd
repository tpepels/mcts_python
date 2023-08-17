#cython: language_level=3
import cython

cdef int win = 9999999
cdef int loss = -9999999
cdef int draw = 0

@cython.freelist(100)
cdef class GameState:
    """
    An abstract base class representing a generic game state.
    """

    # TODO Deze heb je toegevoegd, moet nog in de andere games
    cdef public void apply_action_playout(self, tuple action)

    cpdef public GameState apply_action(self, tuple action)
   
    cdef public tuple get_random_action(self)

    cpdef public list get_legal_actions(self)

    cpdef public bint is_terminal(self)

    cpdef public int get_reward(self, int player)

    # TODO Deze heb je toegevoegd, moet nog in de andere games
    cdef public tuple get_result_tuple(self)

    cdef public bint is_capture(self, tuple move)

    cpdef public list[tuple] evaluate_moves(self, list[tuple] moves)

    cpdef public list move_weights(self, list moves)

    cpdef public int evaluate_move(self, tuple move)

    cdef public double evaluate(self, int function_i, int player, bint norm)


@cython.freelist(10000)
cdef class Move:
    pass

cdef class BTMove(Move):
    cdef int from_position
    cdef int to_position