import re

import numpy as np

from pynnmap.misc import interval_classification as ic

# Module level enumerations
(EQUAL_INTERVAL, QUANTILE, CUSTOM) = range(3)


class VariableVW(object):
    """
    Class to store the values (V) and weights (W) of the variable of
    interest. There is one VariableVW instance for each of the observed
    and predicted data.
    """

    def __init__(self, values, weights):
        """
        Constructor for VariableVW

        Parameters
        ----------
        values : numpy-like array
            Array of values

        weights : numpy-like array
            Array of associated weights, same size as values

        Returns
        -------
        None
        """

        self.values = values
        self.weights = weights


class HistogramBC(object):
    """
    Class to store the bins (B) and counts (C) of data that has been
    binned into histograms.  This should be an abstract super class.
    """

    def __init__(self, bin_counts, bin_endpoints, name):
        """
        Constructor for HistogramBC.  Called from a subclass and populates
        the bin counts, endpoints and series name

        Parameters
        ----------
        bin_counts : numpy array
            One dimensional numpy array of bin counts

        bin_endpoints : numpy array
            One dimensional numpy array of bin endpoints one larger in
            size than bin_counts

        name : string
            Descriptive name for this series

        Returns
        -------
        None
        """

        self.bin_counts = bin_counts
        self.bin_endpoints = bin_endpoints
        self.name = name


class ContinuousHistogramBC(HistogramBC):
    """
    Derived class to store the bins (B) and counts (C) of data
    that has been binned into histograms from continuous data.
    Class names are created from endpoints of the bins
    """

    def __init__(self, bin_counts, bin_endpoints, name="SERIES"):

        # Call the HistogramBC constructor first
        super(ContinuousHistogramBC, self).__init__(
            bin_counts, bin_endpoints, name)

        # Create class labels from the endpoints of the bins
        self.bin_names = []
        if self.bin_endpoints[-1] < 1e5:
            fmt_str = '%.1f'
        else:
            fmt_str = '%.1e'
        for i in range(self.bin_endpoints.size - 1):
            first = fmt_str % (self.bin_endpoints[i])
            second = fmt_str % (self.bin_endpoints[i+1])
            if float(first) == 0.0:
                first = '0.0'
            bn = '-'.join((first, second))
            bn = ''.join(re.split('\+0+', bn))
            self.bin_names.append(bn)


class CategoricalHistogramBC(HistogramBC):
    """
    Derived class to store the bins (B) and counts (C) of data
    that has been binned into histograms from categorical data.
    Class names are passed to the constructor
    """

    def __init__(self, bin_counts, bin_endpoints, bin_names, name="SERIES"):

        # Call the HistogramBC constructor first
        super(CategoricalHistogramBC, self).__init__(
            bin_counts, bin_endpoints, name)

        # Create class labels from the passed argument
        self.bin_names = list(bin_names)


def bin_continuous(datasets, bin_type=EQUAL_INTERVAL, bins=10):
    """
    Given multiple datasets, create histogram bins and counts based on
    the interval classification specified

    Parameters
    ----------
    datasets : list of VariableVW instances
        A list of VariableVW instances which are binned into histogram
        classes.  The weights in the VariableVW instances determine the
        counts in the output histogram classes

    bin_type : enumeration constant
        An enumeration constant.  One of:
        EQUAL_INTERVAL, QUANTILE, CUSTOM

    bins : int
        Number of bins to create.  Defaults to 10

    Returns
    -------
    histogram_data : list of ContinuousHistogramBC instances
    """

    # Create the bin endpoints based on the bin_type requested
    if bin_type == EQUAL_INTERVAL:
        classifier = ic.EqualIntervalClassifier(datasets, bins=bins)
    elif bin_type == QUANTILE:
        classifier = ic.QuantileClassifier(datasets, bins=bins)
    elif bin_type == CUSTOM:
        classifier = ic.CustomIntervalClassifier(bins)

    # Bin the data using this classifier
    histogram_data = []
    for ds in datasets:

        # Create the histogram using the classified endpoints
        counts, endpoints = np.histogram(
            ds.values, bins=classifier.edges, weights=ds.weights)

        # Create a new ContinuousHistogramBC instance and append
        # to histogram_data
        histogram_data.append(ContinuousHistogramBC(counts, endpoints))

    return histogram_data


def bin_categorical(datasets, class_names=None):
    """
    Given multiple datasets, create histogram bins and counts based on
    unique classes present in the data

    Parameters
    ----------
    datasets : list of VariableVW instances
        A list of VariableVW instances which are binned into histogram
        classes.  The weights in the VariableVW instances determine the
        counts in the output histogram classes

    class_names : list
        A list of labels to be matched to the names

    Returns
    -------
    histogram_data : list of CategoricalHistogramBC instances
    """

    # Figure out the unique values in these datasets
    all_unique = []
    for ds in datasets:
        a = np.unique(ds.values)
        all_unique.extend([i for i in a])

    unique = np.unique(all_unique)

    # Create a mapping between these unique values and an enumeration.
    # The enumeration is needed to do a histogram on the data
    class_mapped = {}
    for (i, value) in enumerate(unique):
        class_mapped[value] = i

    # We need to create an ending edge for the last bin.  Because this is an
    # enumeration, we can safely add one to the end of the list for this
    # endpoint.  Note that we only add it to the bin and not to the
    # class_mapped dictionary
    class_values = class_mapped.values()
    class_values.sort()
    class_values.append(max(class_values) + 1)

    # Create the class names either from the class_names variable or from
    # the data themselves.  If coming from the class_names variable, we
    # need to map through the class_mapped keys
    c_names = []
    class_keys = class_mapped.keys()
    class_keys.sort()

    if class_names is not None:
        for key in class_keys:
            if key is not None:
                key = str(key)

                # Find the key in the class_names dict
                value = str(class_names[key])

                # Add this value to the c_names list
                c_names.append(value)
    else:
        for key in class_keys:
            c_names.append(str(key))

    histogram_data = []
    for ds in datasets:

        # Apply the mapping to the original data, in effect creating a
        # lookup table
        enum_data = np.array([class_mapped[x] for x in ds.values])

        # Bin the data using the unique enumeration values
        counts, endpoints = np.histogramdd(
            enum_data, bins=[class_values], weights=ds.weights)

        # Create a new CategoricalHistogramBC instance and append
        # to histogram_data
        histogram_data.append(
            CategoricalHistogramBC(counts, endpoints[0], c_names))

    return histogram_data
