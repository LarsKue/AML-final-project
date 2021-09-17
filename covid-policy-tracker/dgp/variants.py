import numpy as np

from .core import DistributedGaussianProcess
from .partitions import DisjointPartition


class ProductOfExperts(DisjointPartition, DistributedGaussianProcess):
    """
    Join Partitions in a DGP with a product of experts as described in https://arxiv.org/pdf/1502.02843.pdf
    """

    def join(self, means, variances):
        # sigma_k ** -2
        inv_var = 1 / variances
        # sigma_poe ** 2
        var = 1 / np.sum(inv_var, axis=0)
        # mu_poe
        mean = var * np.sum(inv_var * means, axis=0)

        return mean, var
