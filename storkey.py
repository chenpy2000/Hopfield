
import torch
from hopfield_baseline import HopfieldRNN


class HopfieldRNNStorkey(HopfieldRNN):
    def __init__(self, num_units):
        super(HopfieldRNNStorkey, self).__init__(num_units)

    def store_patterns(self, patterns):
        """
        Store patterns in the Hopfield network using the Storkey learning rule.

        Args:
            patterns (torch.Tensor): Binary patterns (shape: [num_patterns, num_units])
        """
        # Initialize weight matrix
        W = torch.zeros(self.num_units, self.num_units, dtype=torch.float32)

        # Apply Storkey learning rule: sum of outer products minus normalization term
        for pattern in patterns:
            outer_prod = torch.outer(pattern, pattern)
            W += outer_prod

        # Normalize weights by the number of patterns
        self.weights.data = W / patterns.shape[0]

        # Storkey adjustment: prevent unbounded weight growth by subtracting
        # the normalization term
        for i in range(self.num_units):
            for j in range(self.num_units):
                sum_term = torch.sum(self.weights[i] * self.weights[j])
                self.weights.data[i, j] = self.weights[i, j] - sum_term / self.num_units