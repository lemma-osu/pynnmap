import numpy as np


class WeightedArray:
    def __init__(self, values, weights):
        values = np.asanyarray(values)
        weights = np.asanyarray(weights)
        if values.shape != weights.shape:
            raise ValueError('Length of values and weights differ')
        self.values = values
        self.weights = weights

    def average(self):
        return np.average(self.values, weights=self.weights)

    def histogram(self, bins):
        return np.histogram(self.values, bins=bins, weights=self.weights)

    def flatten(self):
        return np.hstack(
            [
                np.repeat(self.values[i], np.round(self.weights[i]))
                for i in range(self.values.size)
            ]
        )
