import unittest
from envs.ec.optimal_qmix import OptimalQMIX


class TestOptimalQMIX(unittest.TestCase):

    def test_get_optimal_from_state(self):
        file_path = "/home/csyi/pymarl/src/config/envs/ec.yaml"
        opt_qmix = OptimalQMIX(file_path)
        state = [1, 2.5, 3, 2.5, 1, 2.5, 5, 2.5]
        actions, reward = opt_qmix.get_optimal_from_state(state)
        self.assertEqual(reward, 4.0)
