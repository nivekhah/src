import unittest
from envs.ec.expected_reward import ExpectedReward


class TestExpectedReward(unittest.TestCase):

    def test_get_expected_reward(self):
        """
        [1, 10, 1, 10, 1, 10], 10: 1
        [1, 10, 1, 10, 1, 10], 5: 2
        [1, 10, 1, 10, 3, 10], 10: 2
        :return:
        """
        total_state = [[1, 10, 1, 10, 1, 10],
                       [1, 10, 1, 10, 1, 10],
                       [1, 10, 1, 10, 1, 10],
                       [1, 10, 1, 10, 3, 10],
                       [1, 10, 1, 10, 3, 10]]

        total_reward = [10, 5, 5, 10, 10]

        er = ExpectedReward(total_state, total_reward)
        res = er.get_expected_reward()
        self.assertEqual(res, 8)
