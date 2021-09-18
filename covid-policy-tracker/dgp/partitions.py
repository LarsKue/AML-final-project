import numpy as np


class Partition:
    """
    Partition class used by DistributedGaussianProcess
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Partitioner:
    def partitions(self):
        raise NotImplementedError


class DisjointPartition(Partitioner):
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


class RandomPartition(Partitioner):
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


class CommunicationPartition(Partitioner):
    """
    Base Class for a communication partition in a DistributedGaussianProcess
    Intended for multiple inheritance with another Partitioner
    """
    def __init__(self, x, y, size, **kwargs):
        partition_indices = np.random.choice(len(x), size=size)
        self._communication_partition = Partition(x[partition_indices], y[partition_indices])

        # call the next partitioner (e.g. DisjointPartition)
        super(CommunicationPartition, self).__init__(x, y, **kwargs)

    def partitions(self):
        yield self._communication_partition
        yield from super(CommunicationPartition, self).partitions()
