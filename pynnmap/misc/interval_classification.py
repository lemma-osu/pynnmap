import numpy as np
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


# class IntervalClassifier:
#     """
#     Base class for all interval classifiers
#     """
#     def __init__(self):
#         self.edges = np.array([])
#
#     def __repr__(self):
#         out_str = ''
#         e = self.edges
#         for i in range(e.size - 1):
#             out_str += '({:.4f} - {:.4f})\n'.format(e[i], e[i+1])
#         return out_str


class DynamicClassifier:
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


class EqualIntervalClassifier(DynamicClassifier):
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


class QuantileClassifier(DynamicClassifier):
    """
    Classifier to find class endpoints based on equal bin counts among
    classes
    """

    def __init__(self, bin_count: int = 10):
        super().__init__(bin_count)

    def __call__(self, arr: np.ndarray):
        return approx_quantiles(arr, self.bin_count)


class CustomIntervalClassifier:
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
