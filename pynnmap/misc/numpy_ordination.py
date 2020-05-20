import math

import numpy as np

ZERO = 0.0001


class NumpyCCA(object):
    """
    Calculate canonical correspondence analysis using numpy.  This is a
    pretty direct port of vegan (R) by Jari Oksanen.  See 'cca.default'
    within R for the full code listing.  This class is very stripped down
    when compared with the options in vegan and is meant to be used
    within GNN modeling.

    Note that the variable names are meant to be as consistent as possible
    when compared against vegan
    """

    def __init__(self, x, y):
        """
        Initialize the X (species) and Y (environmental variable) matrices
        and run the ordination.  Key variables are held as properties.
        Note that we usually think of X and Y as the reverse of what is
        presented here, but we want to maintain consistency with vegan

        Parameters
        ----------
        x : np.array
            Species matrix
        y : np.array
            Environment matrix
        """
        self.x = x
        self.y = y

        # Standardize the species matrix
        x = self.x / np.sum(self.x)

        # Ensure that all species rows (plots) have values
        row_sums = np.sum(x, axis=1)
        if np.any(row_sums) <= 0.0:
            err_str = "There were plots with no tally"
            raise ValueError(err_str)

        # Ensure that all species columns (species) have values
        col_sums = np.sum(x, axis=0)
        if np.any(col_sums) <= 0.0:
            err_str = "There were species with no tally"
            raise ValueError(err_str)

        # Compute the outer product of row and column sums
        rc = np.outer(row_sums, col_sums)

        # Chi-square contributions
        x_bar = (x - rc) / np.sqrt(rc)

        # Standardize the environmental matrix based on species row_sums
        self.y_r, weighted_means = self._weight_center(self.y, row_sums)

        # Perform QR decomposition on the weighted Y matrix
        self.q, self.r = np.linalg.qr(self.y_r, mode="full")

        # Run MLR fitting for Y matrix
        right = np.dot(self.q.T, x_bar)
        ls, res, rank, s = np.linalg.lstsq(self.r, right)
        y = np.dot(self.y_r, ls)
        self.u_raw, s, v = np.linalg.svd(y, full_matrices=False)

        # Transform v for later calculations
        v = v.T

        # Determine rank
        if rank > np.sum([s > ZERO]):
            self.rank = np.sum([s > ZERO])
        else:
            self.rank = rank

        # Set instance-level variables for later reporting
        self.eigenvalues = s[0:rank] * s[0:rank]
        self.env_means = weighted_means

        u_weight = np.expand_dims(1.0 / np.sqrt(row_sums), axis=1)
        self.u = np.multiply(self.u_raw[:, 0:rank], u_weight)

        v_weight = np.expand_dims(1.0 / np.sqrt(col_sums), axis=1)
        self.v = np.multiply(v[:, 0:rank], v_weight)

        self.u_eig = np.multiply(self.u, s[0:rank])
        self.v_eig = np.multiply(self.v, s[0:rank])

        a = np.dot(x_bar, v[:, 0:rank])
        self.wa_eig = np.multiply(a, u_weight)

        self.wa = np.multiply(self.wa_eig, (1.0 / s[0:rank]))

    @staticmethod
    def _weight_center(x, w):
        w_c = np.average(x, axis=0, weights=w)
        x = x - w_c
        x = np.apply_along_axis(np.multiply, 0, x, np.sqrt(w))
        return x, w_c

    def biplot_scores(self):
        """
        Return biplot scores of environmental variable by axis
        """
        biplot_scores = np.corrcoef(
            self.y_r, self.u_raw[:, 0 : self.rank], rowvar=False
        )
        return biplot_scores[0 : self.rank, self.rank :]

    def coefficients(self):
        """
        Return the environmental variable loadings of for each axis
        """
        right = np.dot(self.q.T, self.u_raw[:, 0 : self.rank])
        (x, residual, rank, s) = np.linalg.lstsq(self.r, right)
        return np.array(x)

    def species_centroids(self):
        """
        Return the species centroid of each species on each ordination axis
        """
        species_centroids = np.multiply(self.v, np.sqrt(self.eigenvalues))
        return np.array(species_centroids)

    def species_tolerances(self):
        """
        Return species tolerances for each species on each ordination axis
        """
        xi = self.site_lc_scores()
        uk = self.species_centroids()
        xiuk = np.zeros(
            (uk.shape[0], xi.shape[0], xi.shape[1]), dtype=np.float64
        )
        for (i, s) in enumerate(uk):
            xiuk[i] = xi - s
        y = self.x.T
        y_xiuk_sqr = np.zeros((uk.shape[0], uk.shape[1]), dtype=np.float64)
        for i in range(y.shape[0]):
            y_xiuk_sqr[i] = np.dot(y[i], np.square(xiuk[i]))
        species_tolerances = np.sqrt(y_xiuk_sqr / y.sum(axis=1).reshape(-1, 1))
        return species_tolerances

    def species_information(self):
        """
        Return weights and effective number of species occurrences
        """
        species_weights = np.sum(self.x, axis=0)
        a = np.square(np.divide(self.x, species_weights))
        species_n2 = 1.0 / a.sum(axis=0)
        return species_weights, species_n2

    def site_lc_scores(self):
        """
        Return LC scores for each site on each ordination axis.  These scores
        are derived from the linear combination (LC) with the environmental
        variables
        """
        return np.array(self.u)

    def site_wa_scores(self):
        """
        Return WA scores for each site on each ordination axis.  These scores
        are derived from weighted averaging with the species scores
        """
        return np.array(self.wa)

    def site_information(self):
        """
        Return site weights and the effective number of species at a site
        """
        site_weights = np.sum(self.x, axis=1)
        a = np.square(np.divide(self.x, np.expand_dims(site_weights, axis=1)))
        site_n2 = 1.0 / a.sum(axis=1)
        return site_weights, site_n2


class NumpyRDA(object):
    """
    Calculate redundancy analysis using numpy.  This is a
    pretty direct port of vegan (R) by Jari Oksanen.  See 'rda.default'
    within R for the full code listing.  This class is very stripped down
    when compared with the options in vegan and is meant to be used
    within GNN modeling.

    Note that the variable names are meant to be as consistent as possible
    when compared against vegan
    """

    def __init__(self, x, y):
        """
        Initialize the X (species) and Y (environmental variable) matrices
        and run the ordination.  Key variables are held as properties.
        Note that we usually think of X and Y as the reverse of what is
        presented here, but we want to maintain consistency with vegan

        Parameters
        ----------
        x : np.array
            Species matrix
        y : np.array
            Environment matrix
        """
        self.x = x
        self.y = y

        num_rows = x.shape[0] - 1

        col_means = np.mean(x, axis=0).reshape(1, -1)
        x_bar = self.x - col_means
        _, s, _ = np.linalg.svd(x_bar, full_matrices=False)
        self.tot_chi = np.sum(s * s) / num_rows

        col_means = np.mean(y, axis=0).reshape(1, -1)
        self.y_r = self.y - col_means
        self.env_means = col_means.flat

        # Perform QR decomposition on the weighted Y matrix
        self.q, self.r = np.linalg.qr(self.y_r, mode="full")

        # Run MLR fitting for Y matrix
        right = np.dot(self.q.T, x_bar)
        ls, res, rank, s = np.linalg.lstsq(self.r, right)
        y = np.dot(self.y_r, ls)
        u, s, v = np.linalg.svd(y, full_matrices=False)

        # Transform v for later calculations
        v = v.T

        # Determine rank
        if rank > np.sum([s > ZERO]):
            self.rank = np.sum([s > ZERO])
        else:
            self.rank = rank

        # Divide s by degrees of freedom
        s /= math.sqrt(num_rows)

        # Set instance-level variables for later reporting
        self.eigenvalues = s[0:rank] * s[0:rank]

        self.u = u[:, 0:rank]
        self.v = v[:, 0:rank]

        self.u_eig = np.multiply(self.u, s[0:rank])
        self.v_eig = np.multiply(self.v, s[0:rank])

        a = np.dot(x_bar, v[:, 0:rank])
        self.wa_eig = a / math.sqrt(num_rows)

        self.wa = np.multiply(self.wa_eig, (1.0 / s[0:rank]))

    def biplot_scores(self):
        """
        Return biplot scores of environmental variable by axis
        """
        biplot_scores = np.corrcoef(
            self.y_r, self.u[:, 0 : self.rank], rowvar=False
        )
        return biplot_scores[0 : self.rank, self.rank :]

    def coefficients(self):
        """
        Return the environmental variable loadings of for each axis
        """
        right = np.dot(self.q.T, self.u[:, 0 : self.rank])
        (x, residual, rank, s) = np.linalg.lstsq(self.r, right)
        return np.array(x)

    def species_centroids(self):
        """
        Return the species centroid of each species on each ordination axis
        """
        slam = np.sqrt(self.eigenvalues / self.tot_chi)
        v = np.multiply(self.v, slam)
        nr = self.u.shape[0]
        const = np.sqrt(np.sqrt((nr - 1) * self.tot_chi))
        species_centroids = np.multiply(v, const)
        return np.array(species_centroids)

    def species_tolerances(self):
        """
        Return species tolerances for each species on each ordination axis
        """
        xi = self.site_lc_scores()
        uk = self.species_centroids()
        xiuk = np.zeros(
            (uk.shape[0], xi.shape[0], xi.shape[1]), dtype=np.float64
        )
        for (i, s) in enumerate(uk):
            xiuk[i] = xi - s
        y = self.x.T
        y_xiuk_sqr = np.zeros((uk.shape[0], uk.shape[1]), dtype=np.float64)
        for i in range(y.shape[0]):
            y_xiuk_sqr[i] = np.dot(y[i], np.square(xiuk[i]))
        species_tolerances = np.sqrt(y_xiuk_sqr / y.sum(axis=1).reshape(-1, 1))
        return species_tolerances

    def species_information(self):
        """
        Return weights and effective number of species occurrences
        """
        species_weights = np.sum(self.x, axis=0)
        a = np.square(np.divide(self.x, species_weights))
        species_n2 = 1.0 / a.sum(axis=0)
        return species_weights, species_n2

    def site_lc_scores(self):
        """
        Return LC scores for each site on each ordination axis.  These scores
        are derived from the linear combination (LC) with the environmental
        variables
        """
        return np.array(self.u)

    def site_wa_scores(self):
        """
        Return WA scores for each site on each ordination axis.  These scores
        are derived from weighted averaging with the species scores
        """
        return np.array(self.wa)

    def site_information(self):
        """
        Return site weights and the effective number of species at a site
        """
        site_weights = np.sum(self.x, axis=1)
        a = np.square(np.divide(self.x, np.expand_dims(site_weights, axis=1)))
        site_n2 = 1.0 / a.sum(axis=1)
        return site_weights, site_n2
