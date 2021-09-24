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
    def __init__(self, x, y, m, csize=None, **kwargs):
        n = x.shape[0]

        if csize is None:
            # make communication partition same size as other partitions
            csize = int(round(n / m))

        cp_indices = np.random.choice(n, size=csize, replace=False)

        # the communication partition
        cp = Partition(x[cp_indices], y[cp_indices])

        # cp is disjoint with other partitions
        x = np.delete(x, cp_indices, axis=0)
        y = np.delete(y, cp_indices, axis=0)

        n = x.shape[0]

        indices = np.arange(n)
        np.random.shuffle(indices)

        # m - 1 because we already have the communication partition
        partition_indices = np.array_split(indices, m - 1)

        # regular partitions (for training)
        ps = [Partition(x[idx], y[idx]) for idx in partition_indices]

        self._partitions = [
            cp,
            *ps
        ]

        # joined partitions (for prediction)
        jps = [Partition(np.concatenate((cp.x, p.x)), np.concatenate((cp.y, p.y))) for p in ps]

        self._joined_partitions = [
            cp,
            *jps
        ]

        super(CommunicationPartition, self).__init__(**kwargs)

        if not hasattr(self, "training"):
            raise RuntimeError("Communication Partition relies on `self.training` parameter, which was left undefined.")

    def partitions(self):
        if self.training:
            # training process uses unjoined partitions
            return self._partitions
        # prediction process uses joined partitions
        return self._joined_partitions

