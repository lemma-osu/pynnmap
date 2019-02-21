import numpy as np


class VariableVW(object):
    """
    Class to store the values and weights of any variable of interest.
    """

    def __init__(self, values, weights):
        """
        Constructor for VariableVW

        Parameters
        ----------
        values : np.array
            Array of values

        weights : np.array
            Array of associated weights, same size as values

        Returns
        -------
        None
        """

        self.values = values
        self.weights = weights


class IntervalClassifier(object):
    """
    Base class for all interval classifiers
    """

    def __init__(self):
        pass

    def __repr__(self):
        out_str = ''
        for i in range(self.edges.size - 1):
            out_str += \
                '(%.4f' % self.edges[i] + ' - ' + '%.4f)\n' % self.edges[i + 1]
        return out_str


class EqualIntervalClassifier(IntervalClassifier):
    """
    Classifier to find class endpoints based on equal interval splits
    of the range of all datasets
    """

    def __init__(self, datasets, bins=10):
        """
        Set the edges of the classification given one or more datasets and
        the given number of bins.

        The algorithm first determines the range of the classifier based on
        the range present in all the datasets.

        Parameters
        ----------
        datasets : list of array-like objects

        bins : int
            Number of bins

        Returns
        -------
        edges : np.array
            The edges of the classifier.  This list will be one larger than
            the number of bins
        """

        super(EqualIntervalClassifier, self).__init__()

        # Find the absolute min and max of all datasets.  Assume we have
        # VariableVW instances and catch any array-like objects

        try:
            abs_min = min(datasets[0].values)
            abs_max = max(datasets[0].values)
        except AttributeError:
            abs_min = min(datasets[0])
            abs_max = max(datasets[0])

        for ds in datasets:
            try:
                if min(ds.values) < abs_min:
                    abs_min = min(ds.values)
                if max(ds.values) > abs_max:
                    abs_max = max(ds.values)
            except AttributeError:
                if min(ds) < abs_min:
                    abs_min = min(ds)
                if max(ds) > abs_max:
                    abs_max = max(ds)

        # Make sure bins can be cast as an integer
        try:
            bins = int(bins)
        except ValueError:
            raise ValueError('Bins must be a number')

        # Catch negative bins - otherwise they silently succeed
        if bins <= 0:
            raise ValueError('Number of bins must be a positive integer')

        # Create the bins based on these values
        self.edges = numpy.linspace(abs_min, abs_max, num=bins + 1)


class QuantileClassifier(IntervalClassifier):

    def __init__(self, datasets, bins=10):
        super(QuantileClassifier, self).__init__()


class CustomIntervalClassifier(IntervalClassifier):
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

        Returns
        -------
        edges : np.array
            The edges of the classifier.
        """

        super(CustomIntervalClassifier, self).__init__()
        bins = np.array(bins).ravel()
        if bins.size < 1:
            raise ValueError('Bins must not be empty')
        if bins.size == 1:
            bins = np.append(bins, bins[0])
        if (np.diff(bins) < 0).any():
                raise AttributeError('Bins must increase monotonically')
        self.edges = np.array(bins)
