import unittest
from utils.linear_threshold import LinearThreshold

class TestTrial(unittest.TestCase):
    def setup(self):
        self.linear_threshold = LinearThreshold()
    
    def test_infected(self):
        """Test that the number infected <= total nodes in network on a random graph"""
        self.setup()
        infected, N, z = self.linear_threshold.trial(5000, False)
        self.assertLessEqual(infected, N)

    def test_use_all_nodes(self):
        """Tests that all nodes are considered when all == True"""
        self.setup()
        self.assertEqual(5000, self.linear_threshold.trial(5000, True)[1])

    def test_two_nodes(self):
        """Check that everyone is infected if there are only two nodes in the network"""
        self.setup()
        self.assertEqual(2, self.linear_threshold.trial(2, True)[0])

    def test_multi_trials(self):
        """Test the outputs using multiple CPUs are still consistent"""
        self.setup()
        results = self.linear_threshold.multi_trials(5000, False, num_trials = 20, n_cpu = 2)
        for res in results:
            infected, N, z = res
            self.assertLessEqual(infected, N)
