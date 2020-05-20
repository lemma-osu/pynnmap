import re

import numpy as np

from pynnmap.misc import interval_classification as ic
from pynnmap.misc.weighted_array import WeightedArray


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
        bin_counts : np.array
            One dimensional numpy array of bin counts

        bin_endpoints : np.array
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
            bin_counts, bin_endpoints, name
        )

        # Create class labels from the endpoints of the bins
        self.bin_names = []
        if self.bin_endpoints[-1] < 1e5:
            fmt_str = '%.1f'
        else:
            fmt_str = '%.1e'
        for i in range(self.bin_endpoints.size - 1):
            first = fmt_str % (self.bin_endpoints[i])
            second = fmt_str % (self.bin_endpoints[i + 1])
            if float(first) == 0.0:
                first = '0.0'
            bn = '-'.join((first, second))
            bn = ''.join(re.split(r'\+0+', bn))
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
            bin_counts, bin_endpoints, name
        )

        # Create class labels from the passed argument
        self.bin_names = list(bin_names)


def get_dynamic_bins(arr, bin_type="EQUAL_INTERVAL", bins=10):
    """
    Switchyard to get the correct bins to use in classification
    """
    # Create the bin endpoints based on the bin_type requested
    classifier_dict = {
        "EQUAL_INTERVAL": ic.EqualIntervalClassifier,
        "QUANTILE": ic.QuantileClassifier,
    }
    classifier = classifier_dict[bin_type](bins)
    return classifier(arr)


def bin_continuous(*datasets, bins):
    """
    Given multiple datasets, create histogram bins and counts based on
    the interval classification object

    Parameters
    ----------
    datasets : list of VariableVW instances
        A list of VariableVW instances which are binned into histogram
        classes.  The weights in the VariableVW instances determine the
        counts in the output histogram classes

    bins : np.ndarray
        The bins used to classify the array

    Returns
    -------
    histogram_data : list of ContinuousHistogramBC instances
    """
    histogram_data = []
    for ds in datasets:
        if isinstance(ds, WeightedArray):
            counts, endpoints = ds.histogram(bins=bins)
        else:
            counts, endpoints = np.histogram(ds, bins=bins)
        histogram_data.append(ContinuousHistogramBC(counts, endpoints))
    return histogram_data


def bin_categorical(*datasets, class_names=None):
    """
    Given multiple datasets, create histogram bins and counts based on
    unique classes present in the data.  Note that this is only currently
    working for categorical data that are numeric

    Parameters
    ----------
    datasets : sequence of lists, nd-arrays, WeightedArray instances
        The set of datasets to bin
    class_names : list, optional
        A list of labels to be matched to the names

    Returns
    -------
    histogram_data : list of CategoricalHistogramBC instances
    """
    if class_names is None:
        class_names = []

    # Figure out the unique values in these datasets
    all_unique = []
    for ds in datasets:
        if isinstance(ds, WeightedArray):
            a = np.unique(ds.values)
        else:
            a = np.unique(ds)
        all_unique.extend([i for i in a])
    unique = np.unique([int(x) for x in all_unique])

    # Create a mapping between these unique values and an enumeration.
    # The enumeration is needed to do a histogram on the data
    class_mapped = {}
    for (i, value) in enumerate(unique):
        class_mapped[value] = i

    # We need to create an ending edge for the last bin.  Because this is an
    # enumeration, we can safely add one to the end of the list for this
    # endpoint.  Note that we only add it to the bin and not to the
    # class_mapped dictionary
    class_values = list(class_mapped.values())
    class_values.sort()
    class_values.append(max(class_values) + 1)

    # Create the class names either from the class_names variable or from
    # the data themselves.  If coming from the class_names variable, we
    # need to map through the class_mapped keys
    c_names = []
    class_keys = list(class_mapped.keys())
    class_keys.sort()

    if len(class_names):
        for key in class_keys:
            if key is not None:
                key = str(key)

                # Find the key in the class_names dict
                try:
                    value = str(class_names[int(key)])
                except KeyError:
                    value = "Unknown"

                # Add this value to the c_names list
                c_names.append(value)
    else:
        for key in class_keys:
            c_names.append(str(key))

    histogram_data = []
    for ds in datasets:
        # Calculate the histogram through use of a lookup table (class_values)
        if isinstance(ds, WeightedArray):
            mapped = np.array([class_mapped[x] for x in ds.values])
            mapped_ds = WeightedArray(mapped, ds.weights)
            counts, endpoints = mapped_ds.histogram(bins=class_values)
        else:
            mapped = np.array([class_mapped[x] for x in ds])
            counts, endpoints = np.histogram(mapped, bins=class_values)

        # counts, endpoints = np.histogramdd(
        #     enum_data, bins=[class_values], weights=ds.weights)
        # endpoints = list(endpoints)

        # Create a new CategoricalHistogramBC instance and append
        # to histogram_data
        histogram_data.append(
            CategoricalHistogramBC(counts, endpoints, c_names)
        )

    return histogram_data
