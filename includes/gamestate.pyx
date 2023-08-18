#cython: language_level=3
# gamestate.pyx
import cython

@cython.freelist(100)
cdef class GameState:
    """
    An abstract base class representing a generic game state.
    """

    # TODO Deze heb je toegevoegd, moet nog in de andere games
    cdef public void apply_action_playout(self, tuple action):
        pass

    cpdef public GameState apply_action(self, tuple action):
        pass
   
    cdef public tuple get_random_action(self):
        pass

    cpdef public list get_legal_actions(self):
        pass

    cpdef public bint is_terminal(self):
        pass

    cpdef public int get_reward(self, int player):
        pass

    # TODO Deze heb je toegevoegd, moet nog in de andere games
    cdef public tuple get_result_tuple(self):
        pass

    cdef public bint is_capture(self, tuple move):
        pass

    cdef public list[tuple] evaluate_moves(self, list[tuple] moves):
        pass

    cdef public list move_weights(self, list moves):
        pass

    cdef public int evaluate_move(self, tuple move):
        pass

    cdef public double evaluate(self, int function_i, int player, bint norm):
        pass
        