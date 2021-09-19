import GPy
import GPy.models
import numpy as np


class DistributedGaussianProcess(GPy.core.model.Model):
    """
    Base Class for Distributed Gaussian Processes
    Provides basic functionality for distributed predictions and optimization

    Users need to implement how the data partitioning is done and how expert predictions are rejoined.
    """

    name = "DistributedGaussianProcess"

    def __init__(self, kernel=None):
        super(DistributedGaussianProcess, self).__init__(self.name)
        # initialize the gp module and kernel with size of a single partition
        for p in self.partitions():
            self.kernel = kernel or GPy.kern.RBF(input_dim=p.x.shape[1]) + GPy.kern.White(input_dim=p.x.shape[1])
            self._gp = GPy.models.GPRegression(p.x, p.y, kernel=self.kernel)
            break

        print("Partition sizes:", end=" ")
        for p in self.partitions():
            print(p.x.shape, end=" ")
        print()

        self.link_parameters(self._gp)

    def partitions(self):
        raise NotImplementedError

    def join(self, x, means, variances):
        raise NotImplementedError

    def predict(self, x):
        means = []
        variances = []

        for p in self.partitions():
            self._gp.set_XY(p.x, p.y)

            mean, var = self._gp.predict(x)

            means.append(mean)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)

        return self.join(x, means, variances)

    def log_likelihood(self):
        likelihoods = []

        for p in self.partitions():
            self._gp.set_XY(p.x, p.y)

            likelihoods.append(self._gp.log_likelihood())

        return np.sum(likelihoods, axis=0)

    def _log_likelihood_gradients(self):
        gradients = []

        for p in self.partitions():
            self._gp.set_XY(p.x, p.y)

            gradients.append(self._gp._log_likelihood_gradients())

        return np.sum(gradients, axis=0)
