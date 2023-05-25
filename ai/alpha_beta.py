import random
from ai.ai_player import AIPlayer


class AlphaBetaPlayer(AIPlayer):
    def __init__(self, player, depth, evaluate, use_null_moves=False, use_quiescence=False):
        # Identify the player
        self.player = player
        # Set the maximum depth for the search
        self.depth = depth
        # The evaluation function to be used
        self.evaluate = evaluate
        # Whether or not to use the null-move heuristic
        self.use_null_moves = use_null_moves
        # Whether or not to use quiescence search
        self.use_quiescence = use_quiescence
        # Depth reduction for the null-move heuristic
        self.R = 2

    def best_action(self, state):
        def value(state, alpha, beta, depth, null=True):
            max_player = state.player == self.player
            # If the state is terminal, return the reward
            if state.is_terminal():
                return state.get_reward(self.player), None
            # If maximum depth is reached, return the evaluation
            if depth == 0:
                if self.use_quiescence:
                    return self.quiescence(state, alpha, beta), None
                else:
                    return self.evaluate(state, self.player), None

            if max_player:
                # If using null-move heuristic, attempt a null move
                if self.use_null_moves and null and depth >= self.R + 1:
                    null_state = state.skip_turn()
                    null_score, _ = value(null_state, alpha, beta, depth - self.R - 1, False)
                    if null_score >= beta:
                        # print(f"null {null_score=} >= {beta=}")
                        return null_score, None

                # Max player's turn
                v = -float("inf")
                best_move = None
                actions = state.get_legal_actions()
                # Order moves by their heuristic value
                actions.sort(
                    key=lambda move: -state.evaluate_move(move)
                    if self.player == state.player
                    else state.evaluate_move(move)
                )

                for move in actions:
                    new_state = state.apply_action(move)
                    min_v, _ = value(new_state, alpha, beta, depth - 1)
                    if min_v > v:
                        v = min_v
                        best_move = move
                    if v >= beta:
                        return v, move
                    alpha = max(alpha, v)

                if best_move is None:
                    return v, random.sample(actions, 1)[0]

                return v, best_move

            else:  # minimizing player
                v = float("inf")
                actions = state.get_legal_actions()
                # Order moves by heuristic value
                actions.sort(
                    key=lambda move: state.evaluate_move(move)
                    if self.player == state.player
                    else -state.evaluate_move(move)
                )
                for move in actions:
                    new_state = state.apply_action(move)
                    max_v, _ = value(new_state, alpha, beta, depth - 1)
                    if max_v < v:
                        v = max_v
                    if v <= alpha:
                        return v, None
                    beta = min(beta, v)
                return v, None

        # Start the search with the max player
        v, best_move = value(state, -float("inf"), float("inf"), self.depth, True)
        return best_move, v

    def quiescence(self, state, alpha, beta):
        # Quiescence search to avoid the horizon effect
        stand_pat = self.evaluate(state, self.player)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in state.get_legal_actions():
            if state.is_capture(move):
                new_state = state.apply_move(move)
                score = -self.quiescence(new_state, -beta, -alpha)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha
