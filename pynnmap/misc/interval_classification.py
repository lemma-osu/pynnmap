import jenkspy
import numpy as np
import pandas as pd

from pynnmap.misc.weighted_array import WeightedArray


def get_global_range(*datasets):
    """
    Given one or more datasets (either arrays or WeightedArray instances,
    return the global min and max from all datasets

    Parameters
    ----------
    datasets : sequence of lists, nd-arrays, WeightedArray instances
        The set of datasets to use in calculating global min and max

    Returns
    -------
    min : float
        The global min across datasets
    max : float
        The global max across datasets
    """
    if len(list(datasets)) == 0:
        msg = "One of more datasets is needed"
        raise ValueError(msg)
    min_list = []
    max_list = []
    for d in datasets:
        if isinstance(d, WeightedArray):
            d = d.values
        d = np.asanyarray(d)
        min_list.append(d.min())
        max_list.append(d.max())
    return np.array(min_list).min(), np.array(max_list).max()


class RangeIntervals:
    """
    Base class where bin endpoints should represent endpoints used with
    continuous data
    """

    pass


class DynamicIntervals(RangeIntervals):
    """
    Base class for classifiers where endpoints are to be determined.  Only
    the number of bins is specified
    """

    def __init__(self, bin_count):
        try:
            bin_count = int(bin_count)
        except ValueError:
            raise ValueError("Bins must be a number")

        if bin_count <= 0:
            raise ValueError("Number of bins must be a positive integer")
        self.bin_count = bin_count


class EqualIntervals(DynamicIntervals):
    """
    Classifier to find class endpoints based on equal interval splits
    of the range of all datasets
    """

    def __init__(self, bin_count):
        super().__init__(bin_count)

    def __call__(self, arr):
        arr = np.asanyarray(arr)
        return np.linspace(arr.min(), arr.max(), num=self.bin_count + 1)


def approx_quantiles(arr, bin_count):
    if arr.size <= bin_count:
        return np.sort(arr)
    q = np.linspace(0, 1, bin_count + 1)
    bins = np.quantile(arr, q)
    uniq, counts = np.unique(bins, return_counts=True)
    dup = uniq[counts > 1]
    if len(dup):
        new = arr[arr != dup[0]]
        return np.sort(
            np.hstack((dup[0], approx_quantiles(new, bin_count - 1)))
        )
    return bins


class QuantileIntervals(DynamicIntervals):
    """
    Classifier to find class endpoints based on equal bin counts among
    classes
    """

    def __init__(self, bin_count: int = 10):
        super().__init__(bin_count)

    def __call__(self, arr: np.ndarray):
        return approx_quantiles(arr, self.bin_count)


def jenks_natural_breaks(arr, bin_count):
    return jenkspy.jenks_breaks(arr, bin_count)


class NaturalBreaksIntervals(DynamicIntervals):
    """
    Classifier to find class endpoints based on Jenks' natural breaks
    classification
    """

    def __init__(self, bin_count: int = 10):
        super().__init__(bin_count)

    def __call__(self, arr: np.ndarray):
        return jenks_natural_breaks(arr, self.bin_count)


class CustomIntervals(RangeIntervals):
    """
    Classifier to set class endpoints based on user-supplied values
    """

    def __init__(self, bins):
        """
        Set the edges of the classification given user-supplied values

        Parameters
        ----------
        bins : 1-d list
            Bin endpoints
        """
        bins = np.array(bins).ravel()
        if bins.size < 1:
            raise ValueError("Bins must not be empty")
        if bins.size == 1:
            bins = np.append(bins, bins[0])
        if (np.diff(bins) < 0).any():
            raise AttributeError("Bins must increase monotonically")
        self.bins = bins

    def __call__(self, arr):
        return self.bins


class UniqueValues:
    def __call__(self, *arrays):
        uniq = set()
        for arr in arrays:
            uniq |= set(np.unique(arr))
        return np.array(sorted(uniq))


class DataDigitizer:
    PRECISION = 0.0001

    def __init__(self, clf):
        self.clf = clf
        self.bins = None

    def set_bins(self, arr):
        self.bins = self.clf(arr)
        if len(self.bins) == 1:
            self.bins = np.repeat(self.bins, 2)

    def bin_data(self, arr):
        bins_copy = np.array(self.bins, copy=True, dtype=np.float)
        if issubclass(self.clf.__class__, RangeIntervals):
            bins_copy[-1] += self.PRECISION
        else:
            bins_copy = np.append(bins_copy, bins_copy[-1] + self.PRECISION)
        klasses = np.digitize(arr, bins_copy)
        cats = np.arange(1, len(bins_copy))
        return pd.Categorical(klasses, categories=cats)
