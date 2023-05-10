import unittest
from ai.mcts import UCTNode, MCTS


class DummyGameState:
    def __init__(self, player=1, terminal=False, legal_actions=None, reward=0):
        self.player = player
        self.terminal = terminal
        self.legal_actions = legal_actions or []
        self.reward = reward

    def is_terminal(self):
        return self.terminal

    def get_legal_actions(self):
        return self.legal_actions

    def apply_action(self, action):
        return DummyGameState(3 - self.player, terminal=action[1])

    def get_reward(self):
        return self.reward


def dummy_evaluation(state, player):
    return state.reward


class TestUCTNode(unittest.TestCase):
    def test_init(self):
        state = DummyGameState()
        node = UCTNode(state)

        self.assertEqual(node.state, state)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.action)
        self.assertEqual(node.children, [])
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.total_value, 0)
        self.assertFalse(node.solved)
        self.assertIsNone(node.terminal_value)

    def test_add_child(self):
        parent = UCTNode(DummyGameState())
        child = UCTNode(DummyGameState(player=2), parent=parent)

        parent.add_child(child)
        self.assertEqual(parent.children, [child])

    def test_value(self):
        node = UCTNode(DummyGameState())
        self.assertEqual(node.value(), 0)

        node.visits = 4
        node.total_value = 10
        self.assertEqual(node.value(), 2.5)

    def test_uct(self):
        parent = UCTNode(DummyGameState())
        child = UCTNode(DummyGameState(player=2), parent=parent)
        parent.visits = 10

        child.visits = 0
        self.assertEqual(child.uct(1.0), float("inf"))

        child.visits = 4
        child.total_value = 10
        self.assertAlmostEqual(child.uct(1.0), 3.1048, places=4)

    def test_select(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        root = UCTNode(state)
        root.expand()

        for _ in range(10):
            selected_child = root.select(1.0)
            self.assertIn(selected_child, root.children)

    def test_update(self):
        state = DummyGameState()
        node = UCTNode(state)

        node.update(1)
        self.assertEqual(node.visits, 1)
        self.assertEqual(node.total_value, 1)

        node.update(-1)
        self.assertEqual(node.visits, 2)
        self.assertEqual(node.total_value, 0)

    def test_simulate_terminal(self):
        state = DummyGameState(terminal=True, reward=1)
        node = UCTNode(state)

        result = node.simulate()
        self.assertEqual(result, -1)
        self.assertTrue(node.solved)
        self.assertEqual(node.terminal_value, 1)

    def test_simulate_not_expanded(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        node = UCTNode(state)

        result = node.simulate()
        self.assertIsNotNone(result)
        self.assertFalse(node.solved)
        self.assertIsNone(node.terminal_value)
        self.assertEqual(len(node.children), 3)

    def test_simulate_expanded(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        node = UCTNode(state)
        node.expand()

        result = node.simulate()
        self.assertIsNotNone(result)
        self.assertFalse(node.solved)
        self.assertIsNone(node.terminal_value)

    def test_best_action(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        root = UCTNode(state)
        root.expand()

        for child in root.children:
            child.visits = 1
            child.total_value = 0

        best_child = root.children[1]
        best_child.visits = 4
        best_child.total_value = 10

        self.assertEqual(root.best_action(), best_child.action)

    def test_playout(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        node = UCTNode(state)

        playout_value = node.playout(dummy_evaluation)
        self.assertIsNotNone(playout_value)


class TestMCTS(unittest.TestCase):
    def test_init(self):
        state = DummyGameState()
        mcts = MCTS(UCTNode, state, 100)

        self.assertEqual(mcts.node_class, UCTNode)
        self.assertEqual(mcts.state, state)
        self.assertEqual(mcts.num_simulations, 100)
        self.assertEqual(mcts.exploration_param, 1.0)
        self.assertIsInstance(mcts.root, UCTNode)

    def test_run(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        mcts = MCTS(UCTNode, state, 100)
        mcts.run()

        self.assertNotEqual(mcts.root.visits, 0)

    def test_best_action(self):
        state = DummyGameState(legal_actions=[(1, False), (2, False), (3, False)])
        mcts = MCTS(UCTNode, state, 100)
        best_action = mcts.best_action()

        self.assertIn(best_action, [action for action, _ in state.get_legal_actions()])


if __name__ == "__main__":
    unittest.main()
