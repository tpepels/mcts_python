import cython


@cython.ccall
def play_out(state):
    play_out_c(state)


@cython.cfunc
def play_out_c(state):
    while not state.is_terminal():
        best_action = state.get_random_action()
        state.apply_action_playout(best_action)
        print("Action: ", best_action)
        print(state.visualize())

    # Map the result to the players
    print(state.get_result_tuple())
