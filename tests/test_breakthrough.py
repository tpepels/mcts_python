import unittest

from breakthrough import BreakthroughGameState


class TestGameState(unittest.TestCase):
    def test_initial_board(self):
        state = BreakthroughGameState()
        self.assertEqual(state.board[:16], [200 + i for i in range(16)], "Player 2 initial setup incorrect")
        self.assertEqual(state.board[16:48], [0] * 32, "Empty board cells setup incorrect")
        self.assertEqual(state.board[48:], [100 + i for i in range(16)], "Player 1 initial setup incorrect")
        self.assertEqual(state.player, 1, "Initial player should be 1")

    def test_apply_action(self):
        state = BreakthroughGameState()
        action = (48, 40)
        new_state = state.apply_action(action)
        self.assertEqual(new_state.board[48], 0, "Moved piece should be removed from old position")
        self.assertEqual(new_state.board[40], 100, "Moved piece should be at new position")
        self.assertEqual(new_state.player, 2, "Player should change after applying action")

    def test_get_legal_actions_initial_state(self):
        state = BreakthroughGameState()
        legal_actions = state.get_legal_actions()

        # For player 1, only the pieces in the second row should have legal moves.
        self.assertEqual(len(legal_actions), (3 * 6) + (2 * 2))

    def test_get_legal_actions_blocked_pieces(self):
        state = BreakthroughGameState(
            board=[
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
            ],
            player=1,
        )
        legal_actions = state.get_legal_actions()

        # All pieces of player 1 are blocked by their own pieces, so there should be no legal moves.
        self.assertEqual(len(legal_actions), 0)

    def test_get_legal_actions_capture(self):
        state = BreakthroughGameState(
            player=1,
        )
        state.board[21] = 0
        state.board[37] = 101
        legal_actions = state.get_legal_actions()

        # Player 2 has one piece that can capture the opponent's piece.
        capture_actions = [(29, 37)]

        for action in legal_actions:
            if action in capture_actions:
                capture_actions.remove(action)

        self.assertEqual(len(capture_actions), 0)

    def test_get_legal_actions_reach_opponent_home_row(self):
        state = BreakthroughGameState(
            board=[
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                101,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
                202,
            ],
            player=1,
        )
        state.board[15] = 0
        state.board[23] = 101
        legal_actions = state.get_legal_actions()

        # Player 1 has one piece that can reach the opponent's home row.
        reach_opponent_home_row_actions = [(23, 15)]

        for action in legal_actions:
            if action in reach_opponent_home_row_actions:
                reach_opponent_home_row_actions.remove(action)

        self.assertEqual(len(reach_opponent_home_row_actions), 0)

    def test_is_terminal(self):
        state = BreakthroughGameState([0] * 56 + [100 + i for i in range(8)] + [0] * 8)
        self.assertTrue(state.is_terminal(), "Terminal state not detected for player 1")

        state = BreakthroughGameState([200 + i for i in range(8)] + [0] * 56)
        state.player = 2
        self.assertTrue(state.is_terminal(), "Terminal state not detected for player 2")

        state = BreakthroughGameState()
        self.assertFalse(state.is_terminal(), "Initial state should not be terminal")

    def test_get_reward(self):
        state = BreakthroughGameState([0] * 56 + [100 + i for i in range(8)] + [0] * 8)
        self.assertEqual(state.get_reward(), 1, "Reward should be 1 for player 1 winning")

        state = BreakthroughGameState([200 + i for i in range(8)] + [0] * 56)
        state.player = 2
        self.assertEqual(state.get_reward(), -1, "Reward should be -1 for player 2 winning")

        state = BreakthroughGameState()
        self.assertEqual(state.get_reward(), 0, "Reward should be 0 for non-terminal state")


if __name__ == "__main__":
    unittest.main()
