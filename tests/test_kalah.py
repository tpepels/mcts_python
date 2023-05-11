import unittest
from games.kalah import KalahGameState


class TestKalahGameState(unittest.TestCase):
    def test_initial_state(self):
        initial_state = KalahGameState()
        expected_board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        self.assertEqual(initial_state.board, expected_board)
        self.assertEqual(initial_state.player, 1)

    def test_legal_actions_player_1(self):
        initial_state = KalahGameState()
        legal_actions = initial_state.get_legal_actions()
        self.assertEqual(legal_actions, list(range(0, 6)))

    def test_legal_actions_player_2(self):
        state = KalahGameState()
        state = state.apply_action(0)
        legal_actions = state.get_legal_actions()
        self.assertEqual(legal_actions, list(range(7, 13)))

    def test_apply_action_simple_move(self):
        initial_state = KalahGameState.initial_state()
        new_state = initial_state.apply_action((2, 2))
        expected_state = KalahGameState([0, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 0], 2)
        self.assertEqual(new_state, expected_state)

    def test_capture_opponent_seeds(self):
        state = KalahGameState([0, 1, 5, 4, 4, 4, 1, 4, 4, 0, 6, 6, 6, 0], 1)
        state = state.apply_action(1)
        expected_board = [0, 0, 6, 5, 5, 5, 3, 4, 4, 0, 6, 6, 6, 0]
        self.assertEqual(state.board, expected_board)

    def test_no_capture_if_own_house_empty(self):
        state = KalahGameState([0, 0, 0, 5, 5, 5, 2, 5, 4, 0, 6, 6, 6, 0], 1)
        new_state = state.apply_action((1, 8))
        expected_state = KalahGameState([0, 0, 0, 5, 5, 5, 2, 5, 5, 1, 7, 7, 6, 0], 2)
        self.assertEqual(new_state, expected_state)

    def test_terminal_state_player_1_wins(self):
        state = KalahGameState([0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 23], 1)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.get_reward(), 1)

    def test_terminal_state_player_2_wins(self):
        state = KalahGameState([0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 24], 1)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.get_reward(), -1)

    def test_terminal_state_draw(self):
        state = KalahGameState([0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 24], 1)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.get_reward(), 0)


if __name__ == "__main__":
    unittest.main()
