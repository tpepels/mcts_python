from ai.ai_player import AIPlayer


class AlphaBetaPlayer(AIPlayer):
    def __init__(self, state, player, depth, evaluate, use_null_moves=True):
        super().__init__(state)
        self.player = player
        self.depth = depth
        self.evaluate = evaluate
        self.use_null_moves = use_null_moves
        self.R = 2  # Depth reduction for null-move heuristic

    def best_action(self):
        def max_value(state, alpha, beta, depth, null=True):
            if state.is_terminal():
                return state.get_reward(self.player), None
            if depth == 0:
                return self.evaluate(state, self.player), None

            if self.use_null_moves and null and depth >= self.R + 1:
                _, null_move = max_value(state, alpha, beta, depth - self.R - 1, False)
                if null_move >= beta:
                    return null_move, None

            v = -float("inf")
            best_move = None
            for move in state.get_legal_actions():
                new_state = state.apply_move(move)
                min_v, _ = min_value(new_state, alpha, beta, depth - 1)
                if min_v > v:
                    v = min_v
                    best_move = move
                if v >= beta:
                    return v, move
                alpha = max(alpha, v)
            return v, best_move

        def min_value(state, alpha, beta, depth):
            if state.is_terminal():
                return state.get_reward(self.player), None
            if depth == 0:
                return self.evaluate(state, self.player), None

            v = float("inf")
            for move in state.get_legal_actions():
                new_state = state.apply_move(move)
                max_v, _ = max_value(new_state, alpha, beta, depth - 1)
                if max_v < v:
                    v = max_v
                if v <= alpha:
                    return v, None
                beta = min(beta, v)
            return v, None

        _, best_move = max_value(self.state, -float("inf"), float("inf"), self.depth)
        return best_move
