#cython: language_level=3
# gamestate.pxd
import cython

cdef int win, loss, draw

cdef class GameState:
    """
    An abstract base class representing a generic game state.
    """
    cdef public int player

    cdef public void apply_action_playout(self, tuple action)

    cpdef public GameState apply_action(self, tuple action)
   
    cdef public GameState skip_turn(self)

    cdef public tuple get_random_action(self)

    cpdef public list get_legal_actions(self)

    cpdef public bint is_terminal(self)

    cpdef public int get_reward(self, int player) except -1

    cpdef public tuple get_result_tuple(self)

    cdef public bint is_capture(self, tuple move)

    cdef public list[tuple] evaluate_moves(self, list[tuple] moves)

    cdef public list move_weights(self, list moves)

    cdef public int evaluate_move(self, tuple move) except -1

    cdef public double evaluate(self, int player, double[:] params, bint norm=*) except -9999999