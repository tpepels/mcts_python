#cython: language_level=3
# gamestate.pyx
import cython

cdef int win = 9999999
cdef int loss = -9999999
cdef int draw = 0

cdef class GameState:
    """
    An abstract base class representing a generic game state.
    """

    cdef public void apply_action_playout(self, tuple action):
        pass

    cpdef public GameState apply_action(self, tuple action):
        pass
    
    cdef public GameState skip_turn(self):
        pass

    cdef public tuple get_random_action(self):
        pass

    cpdef public list get_legal_actions(self):
        pass

    cpdef public bint is_terminal(self):
        pass

    cpdef public int get_reward(self, short player) except -1:
        pass

    cpdef public tuple get_result_tuple(self):
        pass

    cdef public bint is_capture(self, tuple move):
        pass

    cdef public list[tuple] evaluate_moves(self, list[tuple] moves):
        pass

    cdef public list move_weights(self, list moves):
        pass

    cdef public int evaluate_move(self, tuple move) except -1:
        pass

    cdef double evaluate(self, short player, double[:] params, bint norm=0) except -9999999:
        pass
        