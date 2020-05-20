"""
This module defines a number of useful statistical routines for accuracy
assessment between observed and predicted data sets.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class ZeroSizeError(Exception):
    """
    Zero sized array object
    """


class DataError(Exception):
    """
    Insufficient data for analysis
    """


# Simple least-squares linear regression class including normal, inverse, and
# reduced major axis (rma) predictions
class SimpleLinearRegression:
    def __init__(self, x, y):
        """
        Initialize the object

        Parameters
        ----------
        x : array-like
            Any valid numeric array

        y : array-like
            Any valid numeric array identical in size to x
        """

        self.x = x
        self.y = y
        self.n = x.size

        # Array sums
        self.x_sum = np.sum(self.x)
        self.y_sum = np.sum(self.y)

        # Array means
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)

        # Sum of squares for X, Y and XY
        self.sxx = np.sum((self.x - self.x_mean) * (self.x - self.x_mean))
        self.syy = np.sum((self.y - self.y_mean) * (self.y - self.y_mean))
        self.sxy = np.sum((self.x - self.x_mean) * (self.y - self.y_mean))

    def normal(self):
        """
        Normal regression - minimizing errors in Y
        """

        b1 = self.sxy / self.sxx
        b0 = self.y_mean - (b1 * self.x_mean)
        return b0, b1

    def inverse(self):
        """
        Inverse regression - minimizing errors in X
        """

        y_b1 = self.sxy / self.syy
        y_b0 = self.x_mean - (y_b1 * self.y_mean)
        b1 = 1.0 / y_b1
        b0 = y_b0 / y_b1
        return b0, b1

    def rma(self):
        """
        Reduced major axis regression - minimizing combined errors in X, Y
        """

        b1 = math.sqrt(self.syy / self.sxx)
        b0 = self.y_mean - (b1 * self.x_mean)
        return b0, b1


# Two-by-two error matrix for presence/absence data
class BinaryErrorMatrix:
    def __init__(self, x, y):
        """
        Initialize the object

        Parameters
        ----------
        x : array-like
            Any valid numeric array

        y : array-like
            Any valid numeric array identical in size to x
        """

        # Create the main 2x2 matrix to hold the information
        self._counts = np.zeros((2, 2))

        x_float = _convert_to_float_array(x)
        y_float = _convert_to_float_array(y)

        # Observed present and predicted present
        self._counts[0, 0] = np.logical_and(x_float, y_float).sum()

        # Observed present and predicted absent
        self._counts[0, 1] = np.logical_and(
            x_float, np.logical_not(y_float)
        ).sum()

        # Observed absent and predicted present
        self._counts[1, 0] = np.logical_and(
            np.logical_not(x_float), y_float
        ).sum()

        # Observed absent and predicted absent
        self._counts[1, 1] = np.logical_not(
            np.logical_or(x_float, y_float)
        ).sum()

        # Total number of plots
        self._total = self._counts.sum()

        # Proportions
        self._proportions = self._counts / self._total

        # Store sensitivity and specificity as they are used in other
        # functions
        denominator = self._counts[0, 0] + self._counts[0, 1]
        if denominator == 0.0:
            self._sensitivity = 1.0
        else:
            self._sensitivity = self._counts[0, 0] / denominator

        denominator = self._counts[1, 1] + self._counts[1, 0]
        if denominator == 0.0:
            self._specificity = 1.0
        else:
            self._specificity = self._counts[1, 1] / denominator

    def counts(self):
        return self._counts

    def proportions(self):
        return self._proportions

    def prevalence(self):
        return self._proportions[0, 0] + self._proportions[0, 1]

    def sensitivity(self):
        return self._sensitivity

    def false_negative_rate(self):
        return 1.0 - self._sensitivity

    def specificity(self):
        return self._specificity

    def false_positive_rate(self):
        return 1.0 - self._specificity

    def percent_correct(self):
        return (self._counts[0, 0] + self._counts[1, 1]) / self._total

    def positive_predictive_power(self):
        denominator = self._counts[0, 0] + self._counts[1, 0]
        if denominator == 0.0:
            return 1.0
        else:
            return self._counts[0, 0] / denominator

    def odds_ratio(self):
        denominator = self._counts[0, 1] * self._counts[1, 0]
        if denominator == 0.0:
            return 1.0
        else:
            return (self._counts[0, 0] * self._counts[1, 1]) / denominator

    def kappa(self):
        """
        Cohen's kappa coefficient
        """

        p = self._counts / self._total
        p_chance = ((p[0, 0] + p[0, 1]) * (p[0, 0] + p[1, 0])) + (
            (p[1, 1] + p[0, 1]) * (p[1, 1] + p[1, 0])
        )
        p_correct = p[0, 0] + p[1, 1]

        if p_chance != 1.0:
            kappa = (p_correct - p_chance) / (1.0 - p_chance)
        else:
            kappa = 0.0

        return kappa


def _convert_to_float_array(x):
    """
    Internal function for converting an input array-like parameters to a
    numpy floating-point array

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    Returns
    -------
    out : np.array
        Output numpy array
    """

    x_float = np.array(x, dtype="float64", ndmin=1)
    return x_float


def rmse(x, y):
    """
    Return the root mean square error between two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    out : float
        Root mean square error of x and y

    Example
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> z = rmse(x,y)
    >>> z
    0.57735026918962573
    """

    x_float = _convert_to_float_array(x)
    y_float = _convert_to_float_array(y)

    # Throw an exception for zero-sized arrays
    if x_float.size == 0 or y_float.size == 0:
        raise ZeroSizeError("Input arrays must not have zero elements")

    # Special case when either x_float or y_float is all zero
    if np.all(x_float == 0.0) or np.all(y_float == 0.0):
        return 0.0

    d = x_float - y_float
    return (np.inner(d, d) / len(d)) ** 0.5


def bias_percentage(x, y):
    """
    Return the bias percentage between two numpy arrays.  The mean difference
    is divided by the mean of x (assumed to be the observed array).

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    out : float
        Bias percentage relative to mean of x
    """
    x_float = _convert_to_float_array(x)
    y_float = _convert_to_float_array(y)

    # Throw an exception for zero-sized arrays
    if x_float.size == 0 or y_float.size == 0:
        raise ZeroSizeError("Input arrays must not have zero elements")

    # Special case when either x_float or y_float is all zero
    if np.all(x_float == 0.0) or np.all(y_float == 0.0):
        return 0.0

    return (y_float - x_float).sum() / x_float.sum() * 100.0


def pearson_r(x, y):
    """
    Return the Pearson correlation coefficient between two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    out : float
        Pearson's correlation coefficient between x and y

    Example
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> z = pearson_r(x,y)
    >>> z
    0.98198050606196585
    """
    x_float = _convert_to_float_array(x)
    y_float = _convert_to_float_array(y)

    # Catch insufficient data
    if x_float.size <= 1 or y_float.size <= 1:
        raise DataError("Input arrays need more than one element")

    # Special case when either x_float or y_float is all zero
    if np.all(x_float == 0.0) or np.all(y_float == 0.0):
        return 0.0

    return scipy_stats.pearsonr(x_float, y_float)[0]


def spearman_r(x, y):
    """
    Return the Spearman rank correlation coefficient between two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    out : float
        Spearman's rank correlation coefficient between x and y

    Example
    -------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> z = spearman_r(x,y)
    >>> z
    1.0
    """
    x_float = _convert_to_float_array(x)
    y_float = _convert_to_float_array(y)

    # Catch insufficient data
    if x_float.size <= 1 or y_float.size <= 1:
        raise DataError("Input arrays need more than one element")

    # Special case when either x_float or y_float is all zero
    if np.all(x_float == 0.0) or np.all(y_float == 0.0):
        return 0.0

    return scipy_stats.spearmanr(x_float, y_float)[0]


def r2(x, y):
    """
    Return the coefficient of determination between two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array

    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    out : float
        Coefficient of determination (r2) between x and y

    Example
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> z = r2(x,y)
    >>> z
    0.5
    """
    x_float = _convert_to_float_array(x)
    y_float = _convert_to_float_array(y)

    # Catch insufficient data
    if x_float.size == 1 or y_float.size == 1:
        raise DataError("Input arrays need more than one element")

    # Special case when either x_float or y_float is all zero
    if np.all(x_float == 0.0) or np.all(y_float == 0.0):
        return 0.0

    x_mean = x_float.mean()
    ss_mean = ((x_float - x_mean) * (x_float - x_mean)).sum()
    ss_pred = ((x_float - y_float) * (x_float - y_float)).sum()
    return (ss_mean - ss_pred) / ss_mean


def gmfr(x, y):
    """
    Return the geometric mean functional relationship between two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array
    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    a : float
        Intercept of GMFR relationship
    b : float
        Slope of GMFR relationship
    """
    x_mean = x.mean()
    y_mean = y.mean()
    b = np.sqrt(y.var() / x.var())
    a = y_mean - (b * x_mean)
    return a, b


def ac(x, y):
    """
    Return the agreement coefficient (AC) components for two numpy arrays

    Parameters
    ----------
    x : array-like
        Any valid numeric array
    y : array-like
        Any valid numeric array identical in size to x

    Returns
    -------
    ac : float
        Total agreement coefficient combining systematic and unsystematic
        components
    ac_sys : float
        Systematic component (ie. bias) of the agreement coefficient
    ac_uns : float
        Unsystematic component (ie. precision) of the agreement coefficient
    """
    # Derive the GMFR relationship between these two datasets
    (a, b) = gmfr(x, y)

    # Sum of squared differences
    ssd = ((x - y) * (x - y)).sum()

    # Sum of product differences for GMFR for systematic and unsystematic
    # differences
    c = -a / b
    d = 1.0 / b
    y_hat = a + b * x
    x_hat = c + d * y
    spd_u = (np.abs(x - x_hat) * np.abs(y - y_hat)).sum()
    spd_s = ssd - spd_u

    # Sum of potential differences
    x_mean = x.mean()
    y_mean = y.mean()
    term1 = np.abs(x_mean - y_mean) + np.abs(x - x_mean)
    term2 = np.abs(x_mean - y_mean) + np.abs(y - y_mean)
    spod = (term1 * term2).sum()

    # Agreement coefficients for total, systematic and
    # unsystematic differences
    ac_ = 1.0 - (ssd / spod)
    ac_sys = 1.0 - (spd_s / spod)
    ac_uns = 1.0 - (spd_u / spod)

    return ac_, ac_sys, ac_uns


if __name__ == "__main__":
    import doctest

    doctest.testmod()
