import random
import unittest

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

    def test_random_actions(self):
        state = AmazonsGameState()
        for n in range(100):  # Do 100 random actions on the board
            if state.is_terminal():
                break
            actions = state.get_legal_actions()
            state = state.apply_action(*random.sample(actions, 1))
            board = state.board
            # Check if there are still 8 Amazons
            white_amazons = sum(row.count(1) for row in board)
            black_amazons = sum(row.count(2) for row in board)
            arrows = sum(row.count(-1) for row in board)

            self.assertEqual(
                white_amazons,
                4,
                f"Incorrect number of white Amazons after {n+1} moves. Gamestate: {visualize_amazons(state)}.",
            )
            self.assertEqual(
                black_amazons,
                4,
                f"Incorrect number of black Amazons after {n+1} moves. Gamestate: {visualize_amazons(state)}.",
            )
            self.assertEqual(
                arrows,
                n + 1,
                f"Incorrect number of arrows after {n+1} moves. Gamestate: {visualize_amazons(state)}.",
            )

    def test_is_terminal(self):
        non_terminal_state = AmazonsGameState()
        self.assertFalse(non_terminal_state.is_terminal())

        terminal_board = [[-1] * 10 for _ in range(10)]
        terminal_board[5][3] = terminal_board[5][1] = terminal_board[0][3] = 1
        terminal_board[3][7] = terminal_board[1][7] = terminal_board[3][2] = 2

        terminal_state = AmazonsGameState(terminal_board)
        visualize_amazons(terminal_state)
        self.assertTrue(terminal_state.is_terminal())
        terminal_state.player = 1
        self.assertEqual(terminal_state.get_reward(), -1)
        terminal_state.player = 2
        self.assertEqual(terminal_state.get_reward(), 1)

        terminal_board = [[-1] * 10 for _ in range(10)]
        terminal_state = AmazonsGameState(terminal_board)
        self.assertTrue(terminal_state.is_terminal())

    def test_evaluate(self):
        # TODO Deze moet je nog implementeren
        state = AmazonsGameState()
        eval_value = evaluate(state, 1)
        self.assertIsNotNone(eval_value)

    def test_count_reachable_squares(self):
        # TODO Deze moet je nog implementeren
        state = AmazonsGameState()
        reachable_squares = count_reachable_squares(state, 0, 0)
        self.assertIsNotNone(reachable_squares)


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
