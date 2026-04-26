import numpy as np


class ExampleVectorInterface:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.size = self.rng.integers(2, 10)

    def __next__(self):
        return self.rng.random(self.size)


class ExampleArrayInterface:
    def __init__(self):
        rng = np.random.default_rng()
        self.data = rng.random((rng.integers(2, 10), rng.integers(2, 10)))

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, args):
        if args[1] >= self.shape[1]:
            raise IndexError
        return self.data[:, args[1]]
