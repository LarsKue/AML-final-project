import numpy as np

from .core import DistributedGaussianProcess
from .partitions import DisjointPartition, CommunicationPartition


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
        beta = 0.5 * (np.log(prior) - np.log(variances))

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


class RobustBCM(BayesianCommitteeMachine):
    """
    Robust Bayesian Committee Machine as described in https://arxiv.org/pdf/1502.02843.pdf
    """
    def join(self, x, means, variances):
        prior = np.diag(self._gp.kern.K(x)).reshape(-1, 1)

        inv_var = 1 / variances

        # weights are the same as in CaoFleetPoE
        beta = 0.5 * (np.log(prior) - np.log(variances))

        # sigma_bcm ** 2
        var = 1 / (np.sum(beta * inv_var, axis=0) + (1 - np.sum(beta, axis=0)) / prior)

        # mu_bcm
        mean = var * np.sum(beta * inv_var * means, axis=0)

        return mean, var


class GeneralisedRobustBCM(CommunicationPartition, BayesianCommitteeMachine):
    """
    Generalised Robust Bayesian Committe Machine as described in https://arxiv.org/pdf/1806.00720.pdf
    """
    def join(self, x, means, variances):
        # mu_c
        mean_c = means[0]
        # mu_+i
        means = means[1:]

        # sigma_c ** 2
        var_c = variances[0]
        # sigma_+i ** 2
        variances = variances[1:]

        # beta_i for i = 2...M
        beta = 0.5 * (np.log(var_c) - np.log(variances))
        beta[0] = 1.0

        inv_var_c = 1 / var_c
        inv_var = 1 / variances

        # sigma_A ** 2
        var = 1 / (np.sum(beta * inv_var, axis=0) + (1 - np.sum(beta, axis=0)) * inv_var_c)

        mean = var * (np.sum(beta * inv_var * means, axis=0) + (1 - np.sum(beta, axis=0)) * inv_var_c * mean_c)

        return mean, var





