import itertools
import re
from collections import namedtuple

import numpy as np
import numpy.ma as ma
import pandas as pd
from lxml import objectify


def natural_sort(lst):
    """
    Sorts a list based on natural sorting
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", str(key))]

    return sorted(lst, key=alphanum_key)


def create_error_matrix(obs_data, prd_data, compact=True, classes=None):
    """
    Create an error (confusion) matrix from observed and predicted data.
    The data is assumed to represent classes rather than continuous data.

    Parameters
    ----------
    obs_data : array-like
        Observed classes

    prd_data : array-like
        Predicted classes

    compact : bool
        Flag for whether or not to return error matrix in compact form.  If
        True, only the classes that are represented in the data will be
        returned.  If False, all classes in classes keyword should be
        returned.  Defaults to True.

    classes : array-like
        If compact is False, return error matrix for all classes.  Defaults
        to None.

    Returns
    -------
    err_mat : np.array
        Error matrix of classes

    class_xwalk : dict
        Dictionary of class value to row or column number
    """

    if compact is not True and classes is None or compact is True:
        # No classes given - default to those present as above
        classes = np.union1d(np.unique(obs_data), np.unique(prd_data))
    else:
        # Use the user-defined classes
        classes = np.array(classes)

    n = classes.size

    # One liner for calculating error matrix
    # http://stackoverflow.com/questions/10958702/
    # python-one-liner-for-a-confusion-contingency-matrix-needed
    paired = list(zip(obs_data, prd_data))
    err_mat = np.array(
        [paired.count(x) for x in itertools.product(classes, repeat=2)]
    ).reshape(n, n)

    # Create the dictionary of class value to row/column number
    class_xwalk = dict(zip(classes, range(n)))

    return err_mat, class_xwalk


def kappa(m):
    """
    Calculate kappa for any matrix

    Parameters
    ----------
    m : np.array
        Matrix for which kappa will be calculated

    Returns
    -------
    kappa : float
        Unweighted kappa coefficient for this matrix
    """
    m = m.astype(np.float32)
    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)
    diag_sum = np.diag(m).sum()
    total = m.sum()
    chance = (row_sums * col_sums / total).sum()
    return (diag_sum - chance) / (total - chance) if (total - chance) > 0.0 else 0.0


Classification = namedtuple("Classification", ["value", "name", "fuzzy_values"])


class Classifier:
    """
    Class to provide a categorical classification of a mapped variable
    including information on class values, names and fuzzy classification
    """

    def __init__(self, d):
        # Quick check to make sure we have Classification instances as values
        # in this dict
        for v in d.values():
            if not isinstance(v, Classification):
                err_str = "Classifier contains non-Classification information"
                raise ValueError(err_str)
        self.d = d

    @classmethod
    def from_xml(cls, classifier_file):
        """
        Read classification information from an XML file and return new
        instance

        Parameters
        ----------
        classifier_file : str
            XML classifier file which provides information on values, names
            and fuzzy classification

        Returns
        -------
        cls : Classifier instance
            New instance of Classifier with information for all classes
        """
        tree = objectify.parse(classifier_file)
        root = tree.getroot()
        classifiers = {}
        for i in root.classification:
            value = int(i.value)
            name = str(i.name)
            f_values = [int(j) for j in i.fuzzy_classifications.fuzzy_value]
            classifiers[value] = Classification(value, name, f_values)
        return cls(classifiers)

    @classmethod
    def as_default(cls, values):
        """
        Return a generic Classifier with no fuzzy class information

        Parameters
        ----------
        values : list
            List of values for which to create Classification entries

        Returns
        -------
        cls : Classifier instance
            New instance of Classifier with default information for all classes
        """
        classifiers = {}
        for i in values:
            value = i
            name = f"Class {str(i)}"
            f_values = [i]
            classifiers[value] = Classification(value, name, f_values)
        return cls(classifiers)

    def value(self, i):
        return self.d[i].value

    def name(self, i):
        return self.d[i].name

    def fuzzy_classification(self, i):
        return self.d[i].fuzzy_values

    def values(self):
        return natural_sort(self.d.keys())


class KappaCalculator:
    """
    Class to create and output kappa statistics from an error matrix
    including methods for fuzzy classification
    """

    def __init__(self, obs_data, prd_data, classifier=None):
        """
        Initialize an kappa calculator based on observed and predicted data
        and a specific classifier.  If the classifier is None,
        create a new Classifier instance with no fuzzy information.

        Parameters
        ----------
        obs_data : array_like
            Input array of observed classified values

        prd_data : array_like
            Predicted array of predicted classified values identical
            in size to obs_data

        classifier : Classifier instance
            Instance of a Classifier object.  Defaults to None
        """

        # Set the classifier if present, otherwise initialize a
        # classifier based on data present in the observed and
        # predicted data
        if classifier is None:
            classes = np.union1d(np.unique(obs_data), np.unique(prd_data))
            self.classifier = Classifier.as_default(list(classes))
        else:
            self.classifier = classifier

        # Calculate the error matrix and store in instance variables
        self._calculate(obs_data, prd_data)

    def __repr__(self):
        """
        Create a string representation of this error matrix
        """

        out_str = "CLASS,KAPPA,FUZZY_KAPPA\n"

        # Iterate over classes printing out kappa and fuzzy kappa
        for key in natural_sort(self.kappa_values.keys()):
            if key == "all":
                continue
            v = self.kappa_values[key]
            out_str += f"{key:d},{v['kappa']:.4f},{v['fuzzy']:.4f}\n"

        # Print the values for all classes
        v = self.kappa_values["all"]
        out_str += f"ALL,{v['kappa']:.4f},{v['fuzzy']:.4f}\n"

        return out_str

    def _calculate(self, obs_data, prd_data):
        """
        Calculate kappa and fuzzy kappa statistics for the given data

        Parameters
        ----------
        obs_data : array_like
            Input array of observed classified values

        prd_data : array_like
            Predicted array of predicted classified values identical
            in size to obs_data

        Returns
        -------
        kappa_values : dict
            Dictionary holding kappa and fuzzy kappa values for classes
            defined in the class_xwalk
        """

        # Create an error matrix from the observed and predicted data
        classes = self.classifier.values()
        err_mat, class_xwalk = create_error_matrix(
            obs_data, prd_data, compact=False, classes=classes
        )

        # Get the number of classes
        n_classes = len(class_xwalk)

        # Create a reverse crosswalk as well
        rev_class_xwalk = {i: c for (c, i) in class_xwalk.items()}

        # Create a non-fuzzy mask to apply to the error matrix
        mask = np.diag(np.ones(n_classes))

        # Create a fuzzy classification mask
        f_mask = np.diag(np.ones(n_classes))
        for i in range(n_classes):
            c = rev_class_xwalk[i]
            fcs = self.classifier.fuzzy_classification(c)
            for fc in fcs:
                j = class_xwalk[fc]
                f_mask[i, j] = 1

        # Create the output structure to hold the kappa values
        self.kappa_values = {}

        # For each class, calculate the kappa coefficient
        for i in range(n_classes):
            j = rev_class_xwalk[i]
            self.kappa_values[j] = {}
            self.kappa_values[j]["kappa"] = self._get_masked_kappa(err_mat, mask, c=i)
            self.kappa_values[j]["fuzzy"] = self._get_masked_kappa(err_mat, f_mask, c=i)

        # Calculate kappa for the entire matrix
        self.kappa_values["all"] = {}
        self.kappa_values["all"]["kappa"] = self._get_masked_kappa(err_mat, mask)
        self.kappa_values["all"]["fuzzy"] = self._get_masked_kappa(err_mat, f_mask)

    @staticmethod
    def _get_masked_kappa(err_mat, mask, c=None):
        """
        Given an error matrix and an analysis mask, calculate the kappa
        coefficient.  If c is None, kappa is calculated for the entire matrix,
        otherwise it is calculated for the c specified

        Parameters
        ----------
        err_mat : np.array
            Error (confusion) matrix
        mask : np.array
            Binary analysis mask which specifies what values should be
            considered correct.  Should be the same size as err_mat
        c : int, optional
            Class to calculate.  This corresponds to the row/col of the
            error matrix.  If c is None, kappa is calculated over the entire
            matrix

        Returns
        -------
        kappa : float
            Kappa coefficient for the class or matrix
        """
        # Create a copy of the error matrix as we'll be modifying it below
        err_mat = np.array(err_mat, copy=True)
        em_sum = err_mat.sum()

        # Ensure that the mask is boolean
        mask = np.array(mask, dtype=bool)

        # Single class
        if c is not None:
            # Count the number of cells that are correct according to the
            # mask.  First mask the data, then sum across rows and columns
            # and subtract off the diagonal element to avoid double counting
            m = ma.array(err_mat, mask=~mask)
            op_pp = m[c, :].sum() + m[:, c].sum() - m[c, c]

            # Count the number of cells that are on the row or column of
            # interest but not considered correct.  This is easily done by
            # flipping the mask
            m = ma.array(err_mat, mask=mask)
            op_pa = m[c, :].sum()
            oa_pp = m[:, c].sum()

            # Back calculate the number of cells off the row/col of interest
            oa_pa = em_sum - op_pp - op_pa - oa_pp

            # Create a binary error matrix from this information and get kappa
            m2 = np.array(((op_pp, op_pa), (oa_pp, oa_pa)))
            return kappa(m2)

        # All classes
        #
        # Apply the mask to the data, so that all places where the mask
        # is True, the counts in those off-diagonal bins get reclassified
        # into the diagonal bin.  Do this row-wise as we have symmetry in
        # fuzzy classes.
        for i in range(err_mat.shape[0]):
            for j in range(err_mat.shape[0]):
                if i != j and mask[i, j]:
                    err_mat[i, i] += err_mat[i, j]
                    err_mat[i, j] = 0
        return kappa(err_mat)

    def to_csv(self, kappa_file):
        """
        Prints the kappa statistics to a CSV file

        Parameters
        ----------
        kappa_file : str
            Output file to hold the classification kappa statistics.
            This is output as a comma-separated-value file.

        Returns
        -------
        None
        """
        with open(kappa_file, "w") as kappa_fh:
            kappa_fh.write(self.__repr__())


class ErrorMatrix:
    """
    Class to create and output an error (confusion) matrix including
    methods for fuzzy classification
    """

    def __init__(self, obs_data, prd_data, classifier=None):
        """
        Initialize an error matrix based on observed and predicted data
        and a specific classifier.  If the classifier is None,
        create a new Classifier instance with no fuzzy information.

        Parameters
        ----------
        obs_data : array_like
            Input array of observed classified values

        prd_data : array_like
            Predicted array of predicted classified values identical
            in size to obs_data

        classifier : Classifier instance
            Instance of a Classifier object.  Defaults to None
        """

        # Set the classifier if present, otherwise initialize a
        # classifier based on data present in the observed and
        # predicted data
        if classifier is None:
            classes = np.union1d(np.unique(obs_data), np.unique(prd_data))
            self.classifier = Classifier.as_default(list(classes))
        else:
            self.classifier = classifier

        # Calculate the error matrix and store in instance variables
        self._calculate(obs_data, prd_data)

    def __repr__(self):
        """
        Create a string representation of this error matrix
        """

        # Short circuit the condition where the error matrix has not been
        # calculated
        if self.err_mat is None:
            return ""

        # Labels for x and y axes
        class_labels = [repr(x) for x in natural_sort(self.class_xwalk.keys())]
        class_labels.extend(["Total", "% Correct", "% FCorrect"])

        # Print out these stats
        # Column labels
        out_str = ""
        out_str += "," + ",".join(class_labels) + "\n"
        num_classes = len(self.class_xwalk)

        # Row labels, values, totals, percent correct and percent fuzzy correct
        for i in range(num_classes):
            out_list = [
                class_labels[i],
                *["%d" % x for x in self.err_mat[i, :]],
                "%d" % self.r_totals[i],
                "%.3f" % self.r_percent_correct[i],
                "%.3f" % self.r_percent_f_correct[i],
            ]
            out_str += ",".join(out_list) + "\n"

        # Column totals
        out_list = [
            class_labels[num_classes],
            *["%d" % x for x in self.c_totals],
            *["%d" % self.m_total, "", ""],
        ]
        out_str += ",".join(out_list) + "\n"

        # Column correct
        out_list = [
            class_labels[num_classes + 1],
            *["%.3f" % x for x in self.c_percent_correct],
            *["", "%.3f" % self.m_percent_correct, ""],
        ]
        out_str += ",".join(out_list) + "\n"

        # Column fuzzy correct
        out_list = [
            class_labels[num_classes + 2],
            *["%.3f" % x for x in self.c_percent_f_correct],
            *["", "", "%.3f" % self.m_percent_f_correct],
        ]
        out_str += ",".join(out_list) + "\n"

        return out_str

    def _calculate(self, obs_data, prd_data):
        """
        Calculate an classification error matrix based on observed and
        predicted data and store results within instance variables

        Parameters
        ----------
        obs_data : array_like
            Input array of observed classified values

        prd_data : array_like
            Predicted array of predicted classified values identical
            in size to obs_data

        Returns
        -------
        None
        """

        # Create an error matrix from the observed and predicted data
        classes = self.classifier.values()
        self.err_mat, self.class_xwalk = create_error_matrix(
            obs_data, prd_data, compact=False, classes=classes
        )

        # Number of classes
        num_classes = len(self.class_xwalk)

        # Get the total row, column and matrix counts
        self.r_totals = self.err_mat.sum(axis=1).astype(np.float64)
        self.c_totals = self.err_mat.sum(axis=0).astype(np.float64)
        self.m_total = self.err_mat.sum().astype(np.float64)

        # Get the diagonal elements - this represents the plots classified
        # correctly
        self.correct = np.diag(self.err_mat).astype(np.float64)

        # Calculate the fuzzy correct for each class
        self.r_f_correct = np.zeros(num_classes)
        self.c_f_correct = np.zeros(num_classes)
        self.m_f_incorrect = 0

        for c, i in sorted(self.class_xwalk.items()):
            # Get the fuzzy classes and indexes associated with this class
            f_classes = self.classifier.fuzzy_classification(c)
            f_indexes = np.array([self.class_xwalk[x] for x in f_classes])

            # Get the number of fuzzy correct for this index's row and column
            self.r_f_correct[i] = self.err_mat[i, f_indexes].sum()
            self.c_f_correct[i] = self.err_mat[f_indexes, i].sum()

            # Find total fuzzy incorrect indirectly by counting those elements
            # that are correct and subtracting this off the total number of
            # plots - only need to do this for rows to avoid double counting
            self.m_f_incorrect += self.r_totals[i] - self.r_f_correct[i]

        # Calculate the percentages used in the printed error matrix
        # Temporarily suspend warnings
        old_settings = np.seterr(all="ignore")

        def calc_percent(n, d):
            return np.where(d, n / d * 100.0, 0.0)

        self.r_percent_correct = calc_percent(self.correct, self.r_totals)
        self.c_percent_correct = calc_percent(self.correct, self.c_totals)
        self.m_percent_correct = calc_percent(self.correct.sum(), self.m_total)

        self.r_percent_f_correct = calc_percent(self.r_f_correct, self.r_totals)
        self.c_percent_f_correct = calc_percent(self.c_f_correct, self.c_totals)
        m_f_correct = self.m_total - self.m_f_incorrect
        self.m_percent_f_correct = calc_percent(m_f_correct, self.m_total)
        np.seterr(**old_settings)

    def to_csv(self, err_matrix_fn):
        """
        Prints the error matrix to a CSV file

        Parameters
        ----------
        err_matrix_fn : str
            Output file to hold the classification error matrix.
            This is output as a comma-separated-value file.

        Returns
        -------
        None
        """
        with open(err_matrix_fn, "w") as err_matrix_fh:
            err_matrix_fh.write(self.__repr__())


def print_kappa_file(obs_data, prd_data, classifier, kappa_fn):
    """
    Calculate kappa and fuzzy kappa statistics and print to an output file

    Parameters
    ----------
    obs_data : array_like
        Input array of observed classified values

    prd_data : array_like
        Predicted array of predicted classified values identical
        in size to obs_data

    classifier : Classifier instance
        Instance of a Classifier object corresponding to
        the classes present in obs_data and prd_data

    kappa_fn : str
        Output file to hold the kappa and fuzzy kappa statistics.
        This is output as a comma-separated-value file.

    Returns
    -------
    None
    """

    # Create a Kappa instance
    k = KappaCalculator(obs_data, prd_data, classifier=classifier)

    # Write out the error matrix as an object
    k.to_csv(kappa_fn)


def print_error_matrix_file(obs_data, prd_data, classifier, err_matrix_fn):
    """
    Calculate an classification error matrix and print to an output file

    Parameters
    ----------

    obs_data : array_like
        Input array of observed classified values

    prd_data : array_like
        Predicted array of predicted classified values identical
        in size to obs_data

    classifier : Classifier instance
        Instance of a Classifier object corresponding to the classes present
        in obs_data and prd_data

    err_matrix_fn : str
        Output file to hold the classification error matrix.
        This is output as a comma-separated-value file.

    Returns
    -------
    None
    """

    # Create an ErrorMatrix object
    e = ErrorMatrix(obs_data, prd_data, classifier=classifier)

    # Write out the error matrix as an object
    e.to_csv(err_matrix_fn)


def classification_accuracy(
    input_fn,
    classifier_fn,
    kappa_fn=None,
    err_matrix_fn=None,
    observed_column="OBSERVED",
    predicted_column="PREDICTED",
):
    """
    Wrapper function to read in a plot-by-classification file
    of observed and predicted values and a classifier XML file
    and return output kappa statistics and error matrix for a
    given variable

    Parameters
    ----------
    input_fn : str
        The input filename (comma-separated-value format) with the
        observed and predicted classified values for all plots.
        The file must have a header line with column names.
        Specify the names for the observed and predicted
        columns using the 'observed_column' and 'predicted_column'
        keyword parameters.

    classifier_fn : str
        An XML filename that describes the variable classification
        including information on fuzzy sets.  This file must
        validate against 'classifier.xsd'.

    kappa_fn : str
        Output filename to hold kappa and fuzzy kappa statistics.
        Defaults to None (ie. not output).

    err_matrix_fn : str
        Output filename to hold error matrix statistics.
        Default to None (ie. not output).

    observed_column : str
        The name of the observed column in the input_file.
        Defaults to 'OBSERVED'

    predicted_column : str
        The name of the predicted column in the input_file.
        Defaults to 'PREDICTED'

    Returns
    -------
    None
    """

    # Read in the raw input file
    csv = pd.read_csv(input_fn)
    obs_data = csv[observed_column]
    prd_data = csv[predicted_column]

    # Read in the classification
    c = Classifier.from_xml(classifier_fn)

    # Print classification kappas
    if kappa_fn is not None:
        print_kappa_file(obs_data, prd_data, c, kappa_fn)

    # Print classification error matrix
    if err_matrix_fn is not None:
        print_error_matrix_file(obs_data, prd_data, c, err_matrix_fn)
