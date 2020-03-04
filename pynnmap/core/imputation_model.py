import numpy as np
from sklearn.neighbors import NearestNeighbors


class ImputationModel(object):

    def __init__(self, ord_model, n_axes=8, use_weightings=True,
                 max_neighbors=100):

        # Ensure that n_axes isn't larger than the number of axes in our
        # ordination model
        self.n_axes = n_axes
        if self.n_axes > ord_model.n_axes:
            self.n_axes = ord_model.n_axes

        # Ensure that max_neighbors isn't larger than the number of plots
        # in our ordination model
        self.max_neighbors = max_neighbors
        if self.max_neighbors > ord_model.n_plots:
            self.max_neighbors = ord_model.n_plots

        # Create weightings based on use_weightings flag
        if use_weightings:
            self.ax_weights = \
                np.diag(np.sqrt(ord_model.axis_weights[0:n_axes]))
        else:
            self.ax_weights = np.diag(np.ones(n_axes, dtype=np.float))

        # Set up the ANN model
        # Use the parameter 'max_neighbors' to determine how many neighbors
        # to return
        plot_scores = \
            np.dot(ord_model.plot_scores[:, 0:n_axes], self.ax_weights)
        self.nn_finder = NearestNeighbors(max_neighbors)
        self.nn_finder.fit(plot_scores)

        # Create arrays of the ord_model's var_coeff and axis_intercepts
        # for repeated use
        self.var_coeff = ord_model.var_coeff
        self.axis_intercepts = ord_model.axis_intercepts

        # Lookup of index to plot ID to get correct neighbor IDs
        self.ipd = ord_model.id_plot_dict

    def get_neighbors(self, env_values, id_val=None):
        """
        Given the vector of env_values, return the sorted neighbors and
        distances for this vector.  If id is specified, ensure that if the
        id "ties" for the nearest neighbor, it is returned first in the list

        Parameters
        ----------
        env_values : np.array
            A vector of environmental values (1 x v) for which to determine
            neighbors and distances

        id_val : int
            The ID of this vector, if known.  This ID is guaranteed to be
            ordered as first if it is the nearest neighbor

        Returns
        -------
        neighbor_ids : np.array
            Sorted vector of nearest neighbor IDs

        distance_arr : np.array
            Sorted vector of nearest neighbor distances
        """

        # Transform raw environmental scores to ANN space by multiplying by
        # the coefficient matrix and then the ax_weight matrix
        axis_scores = np.dot(env_values, self.var_coeff) - self.axis_intercepts
        axis_scores = axis_scores[:, 0:self.n_axes]
        axis_scores = np.dot(axis_scores, self.ax_weights)

        # Run the imputation
        d, n = self.nn_finder.kneighbors(axis_scores)
        d, n = d[0], n[0]

        # If id is specified, find ties in distances - this happens when
        # two plots have exactly the same environmental information. Swap
        # neighbors such that ID is ordered first
        if id_val is not None:
            same_distances = np.where(d == d[0])[0]
            if len(same_distances) > 1:
                for i in range(1, len(same_distances)):
                    if self.ipd[n[i]] == id_val:
                        temp = n[0]
                        n[0] = n[i]
                        n[i] = temp

        # Crosswalk the neighbors from indexes to IDs
        neighbor_ids = np.array([self.ipd[x] for x in n])

        # Return neighbor IDs and distances
        return neighbor_ids, d
