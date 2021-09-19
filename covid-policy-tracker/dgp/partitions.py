import numpy as np


class Partition:
    """
    Partition class used by DistributedGaussianProcess
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DisjointPartition:
    """
    Base Class for Disjoint Partitioning in a DistributedGaussianProcess
    """
    def __init__(self, x, y, m, **kwargs):
        n = x.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)

        partition_indices = np.array_split(indices, m)
        self._partitions = [Partition(x[idx], y[idx]) for idx in partition_indices]

        # call this last to ensure self._partitions is defined
        super(DisjointPartition, self).__init__(**kwargs)

    def partitions(self):
        return self._partitions


class RandomPartition:
    """
    Base Class for Random Partitioning in a DistributedGaussianProcess
    """
    def __init__(self, x, y, m, **kwargs):
        n = x.shape[0]
        partition_indices = np.random.choice(n, size=(m, n), replace=True)
        self._partitions = [Partition(x[idx], y[idx]) for idx in partition_indices]

        # call this last to ensure self._partitions is defined
        super(RandomPartition, self).__init__(**kwargs)

    def partitions(self):
        return self._partitions


class CommunicationPartition:
    def __init__(self, x, y, csize, m, **kwargs):
        n = x.shape[0]
        cp_indices = np.random.choice(n, size=csize)
        cp = Partition(x[cp_indices], y[cp_indices])

        # cp is disjoint with other partitions
        x = np.delete(x, cp_indices, axis=0)
        y = np.delete(y, cp_indices, axis=0)

        n = x.shape[0]

        indices = np.arange(n)
        np.random.shuffle(indices)

        partition_indices = np.array_split(indices, m)

        p = [Partition(np.concatenate((cp.x, x[idx])), np.concatenate((cp.y, y[idx]))) for idx in partition_indices]

        self._partitions = [
            cp,
            *p
        ]

        super(CommunicationPartition, self).__init__(**kwargs)

    def partitions(self):
        return self._partitions
