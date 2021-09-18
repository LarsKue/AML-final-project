import numpy as np

from .core import DistributedGaussianProcess
from .partitions import DisjointPartition


class ProductOfExperts(DisjointPartition, DistributedGaussianProcess):
    """
    Join Partitions in a DGP with a product of experts as described in https://arxiv.org/pdf/1502.02843.pdf
    """
    def join(self, x, means, variances):
        # sigma_k ** -2
        inv_var = 1 / variances
        # sigma_poe ** 2
        var = 1 / np.sum(inv_var, axis=0)
        # mu_poe
        mean = var * np.sum(inv_var * means, axis=0)

        return mean, var


class CaoFleetPoE(ProductOfExperts):
    """
    Generalised Product of Experts with weights for each expert as described in https://arxiv.org/pdf/1410.7827.pdf
    """
    def join(self, x, means, variances):
        prior = np.diag(self._gp.kern.K(x)).reshape(-1, 1)

        # change in entropy from prior to posterior
        beta = 0.5 * np.abs(np.log(2 * np.pi * variances) - np.log(2 * np.pi * prior))

        # sigma_k ** -2
        inv_var = 1 / variances
        # sigma_poe ** 2
        var = 1 / np.sum(beta * inv_var, axis=0)
        # mu_poe
        mean = var * np.sum(beta * inv_var * means, axis=0)

        return mean, var


class GeneralisedPoE(ProductOfExperts):
    """
    Generalised Product of Experts with identical weights for each expert as described in https://arxiv.org/pdf/1502.02843.pdf
    """
    def join(self, x, means, variances):
        # sigma_k ** -2
        inv_var = 1 / variances
        # sigma_poe ** 2
        var = 1 / np.mean(inv_var, axis=0)
        # mu_poe
        mean = var * np.mean(inv_var * means, axis=0)

        return mean, var


class BayesianCommitteeMachine(DisjointPartition, DistributedGaussianProcess):
    """
    Bayesian Committee Machine as described in https://arxiv.org/pdf/1502.02843.pdf
    """
    def join(self, x, means, variances):
        prior = np.diag(self._gp.kern.K(x)).reshape(-1, 1)

        M = len(means)

        # sigma_k ** -2
        inv_var = 1 / variances
        # sigma_bcm ** 2
        var = 1 / (np.sum(inv_var, axis=0) + (1 - M) / prior)
        # mu_bcm
        mean = var * np.sum(inv_var * means, axis=0)

        return mean, var

