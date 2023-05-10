import unittest
import math

from games.amazons import AmazonsGameState, evaluate, count_reachable_squares, visualize_amazons


class TestAmazonsGameState(unittest.TestCase):
    def _assert_board(self, board, expected_board, message=None):
        # Check if there are still 8 Amazons
        white_amazons = sum(row.count(1) for row in board)
        black_amazons = sum(row.count(2) for row in board)
        self.assertEqual(white_amazons, 4, f"Incorrect number of white Amazons. {message}")
        self.assertEqual(black_amazons, 4, f"Incorrect number of black Amazons. {message}")

        for i in range(10):
            for j in range(10):
                self.assertEqual(
                    board[i][j], expected_board[i][j], f"Board position ({i}, {j}) is incorrect. {message}"
                )

    def test_initialize_board(self):
        state = AmazonsGameState()
        board = state.initialize_board()
        init_board = [
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
        ]

        self._assert_board(board, expected_board=init_board, message="Wrong board initialized.")

    def test_apply_action(self):
        state = AmazonsGameState()
        action = (0, 3, 8, 3, 2, 3)
        new_state = state.apply_action(action)

        expected_board = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
        ]

        self._assert_board(
            new_state.board, expected_board=expected_board, message=f"Wrong state after move {str(action)}."
        )

        self.assertNotEqual(state.player, new_state.player)

    def test_get_legal_actions(self):
        state = AmazonsGameState()
        actions = state.get_legal_actions()

        self.assertTrue(len(actions) > 0)
        self.assertTrue((0, 0, 1, 1, 1, 2) in actions)

    def test_is_terminal(self):
        state = AmazonsGameState()
        self.assertFalse(state.is_terminal())

        terminal_board = [[-1] * 10 for _ in range(10)]
        terminal_state = AmazonsGameState(terminal_board)
        self.assertTrue(terminal_state.is_terminal())

    def test_get_reward(self):
        state = AmazonsGameState()
        self.assertEqual(state.get_reward(), 0)

        terminal_board = [[-1] * 10 for _ in range(10)]
        terminal_state = AmazonsGameState(terminal_board)
        self.assertEqual(terminal_state.get_reward(), 1)

        terminal_state_opponent = AmazonsGameState(terminal_board, player=2)
        self.assertEqual(terminal_state_opponent.get_reward(), -1)

    def test_evaluate(self):
        state = AmazonsGameState()
        eval_value = evaluate(state, 1)
        self.assertIsNotNone(eval_value)

    def test_count_reachable_squares(self):
        state = AmazonsGameState()
        reachable_squares = count_reachable_squares(state, 0, 0)
        self.assertIsNotNone(reachable_squares)

    def test_visualize_amazons(self):
        state = AmazonsGameState()
        try:
            visualize_amazons(state)
            success = True
        except Exception:
            success = False

        self.assertTrue(success)


# class TestHeuristic(unittest.TestCase):
#     def setUp(self):
#         self.board = [
#             # TODO Fill in your test board state
#         ]
#         self.is_white_player = True
#         self.max_depth = 5
#         self.heuristic = Heuristic(self.board, self.is_white_player, self.max_depth)

#     def test_kill_save_queens(self):
#         result = self.heuristic.kill_save_queens()
#         expected_result = 0  # TODO Set the expected result based on your test board state
#         self.assertEqual(result, expected_result)

#     def test_immediate_moves_heuristic(self):
#         result = self.heuristic.immediate_moves_heuristic()
#         expected_result = 0  # TODO Set the expected result based on your test board state
#         self.assertEqual(result, expected_result)

#     def test_mobility_heuristic(self):
#         result = self.heuristic.mobility_heuristic()
#         expected_result = 0  # TODO Set the expected result based on your test board state
#         self.assertEqual(result, expected_result)

#     def test_territory_heuristic(self):
#         result = self.heuristic.territory_heuristic()
#         expected_result = 0  # TODO Set the expected result based on your test board state
#         self.assertEqual(result, expected_result)

#     def test_territory_helper(self):
#         out = [[math.inf] * self.heuristic.N for _ in range(self.heuristic.N)]
#         queen_position = (2, 2)  # TODO Set a queen position for the test
#         self.heuristic.territory_helper(queen_position, out)

#         # Set the expected result based on your test board state and queen_position
#         expected_out = [
#             # TODO Fill in the expected territory matrix
#         ]
#         self.assertEqual(out, expected_out)

#     def test_territory_compare(self):
#         ours = [
#             # TODO Fill in the test 'ours' territory matrix
#         ]
#         theirs = [
#             #  TODO Fill in the test 'theirs' territory matrix
#         ]

#         result = self.heuristic.territory_compare(ours, theirs)
#         expected_result = 0  # TODO Set the expected result based on your test matrices
#         self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
