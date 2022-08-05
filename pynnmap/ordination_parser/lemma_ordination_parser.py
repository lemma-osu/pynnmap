import re
import sys

import numpy as np

from pynnmap.core import ordination_model
from pynnmap.misc import parser


class LemmaOrdinationParser(parser.Parser):
    def __init__(self, delimiter=","):
        """
        Parameters
        ----------
        delimiter : str, optional
            Delimiter that separates fields in data sections
        """
        super(LemmaOrdinationParser, self).__init__()
        self.delimiter = delimiter

    def parse(self, ordination_file):
        """
        Parse an ordination file and return an OrdinationModel to the caller.
        Currently, this handles both canonical correspondence analysis (CCA)
        and redundancy analysis (RDA) ordination types.

        Parameters
        ----------
        ordination_file : str
            Name of the ordination file

        Returns
        -------
        model : OrdinationModel instance
        """
        with open(ordination_file, "r") as ordination_fh:
            all_lines = ordination_fh.readlines()
        # Create an empty OrdinationModel
        model = ordination_model.OrdinationModel()

        # Axis weights derived from eigenvalues
        model.axis_weights = self._get_axis_weights(all_lines)

        # Variable coefficients and names
        model.var_names, model.var_coeff = self._get_coefficients(all_lines)

        # Variable means and names
        var_names_1, means = self._get_means(all_lines)

        # Species centroids and names
        model.species_names, model.species_scores = self._get_species_scores(
            all_lines
        )

        # Plot scores and IDs
        model.plot_ids, model.plot_scores = self._get_plots(all_lines)

        # Variable biplot scores and names
        var_names_2, model.biplot_scores = self._get_biplot_scores(all_lines)

        # Data checks
        try:
            parser.assert_same_set(var_names_1, model.var_names)
            parser.assert_same_set(var_names_2, model.var_names)
        except parser.ParserError:
            err_msg = "Variable names are not the same in all sections"
            raise parser.ParserError(err_msg)

        try:
            parser.assert_same_size(model.axis_weights, model.var_coeff[0])
        except parser.ParserError:
            err_msg = "Number of axes differ between eigenvalues " + "and coefficients"
            raise parser.ParserError(err_msg)

        try:
            parser.assert_same_size(model.axis_weights, model.plot_scores[0])
        except parser.ParserError:
            err_msg = "Number of axes differ between eigenvalues " + "and plot scores"
            raise parser.ParserError(err_msg)

        try:
            parser.assert_same_size(model.axis_weights, model.biplot_scores[0])
        except parser.ParserError:
            err_msg = "Number of axes differ between eigenvalues and " + "biplot scores"
            raise parser.ParserError(err_msg)

        # Set model parameter counts
        model.n_variables = len(model.var_names)
        model.n_axes = model.axis_weights.size
        model.n_species = model.species_names.size
        model.n_plots = len(model.plot_ids)

        # Calculate axis intercepts from means, coefficients
        model.axis_intercepts = np.dot(means, model.var_coeff)

        # Create dictionaries of plot_ids to index and var_names to index
        model.plot_id_dict = {}
        model.id_plot_dict = {}
        for (i, plot_id) in enumerate(model.plot_ids):
            model.plot_id_dict[plot_id] = i
            model.id_plot_dict[i] = plot_id

        model.var_name_dict = {
            var_name: i for i, var_name in enumerate(model.var_names)
        }

        # Return the model to the caller
        return model

    def _get_axis_weights(self, all_lines):
        """
        Read in axis_weights (eigenvalues) from ordination file and return
        as a 1 x n_axes array

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        axis_weights : np.array
            Axis weights (based on eigenvalues) from ordination model
        """

        # Get the lines associated with the model eigenvalues
        eig_re = re.compile(r"^###\s+Eigenvalues\s+###.*")
        chunks = self.read_chunks(
            all_lines, eig_re, self.blank_re, skip_lines=1, flush=True
        )

        # Extract the axis weights
        axis_weights = []
        for chunk in chunks:
            for line in chunk:
                axis_weight = line.strip().split(self.delimiter)[1]
                axis_weights.append(float(axis_weight))
        return np.array(axis_weights)

    def _get_coefficients(self, all_lines):
        """
        Read in variable coefficients from ordination file (the result of
        multiple linear regression fit) and return variable names as a
        1 x n_variables array and variable coefficients as a n_variables
        x n_axes array

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        var_names : np.array
            Variable names associated with the ordination variables

        coefficients : np.array
            Coefficients for each variable and axis from ordination model
        """

        # Get the lines associated with the model coefficients
        coeff_re = re.compile(r"^###\s+Coefficient\s+Loadings\s+###")
        chunks = self.read_chunks(
            all_lines, coeff_re, self.blank_re, skip_lines=2, flush=True
        )

        # Extract the variable names and coefficients
        var_names = []
        coefficients = []
        for chunk in chunks:
            for line in chunk:
                data = line.strip().split(self.delimiter)
                var_names.append(data[0])
                coefficients.append([float(x) for x in data[1:]])
        return np.array(var_names), np.array(coefficients)

    def _get_means(self, all_lines):
        """
        Read in variable means from ordination file and return variable names
        as a 1 x n_variables array and variable means as a 1 x n_variables
        array

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        var_names : np.array
            Variable names associated with the ordination variables
        var_means : np.array
            Means of the ordination variables
        """

        # Get the lines associated with the ordination variable means
        mean_re = re.compile(r"^###\s+Variable\s+Means\s+###.*")
        chunks = self.read_chunks(
            all_lines, mean_re, self.blank_re, skip_lines=1, flush=True
        )

        # Extract the variable names and means
        var_names = []
        var_means = []
        for chunk in chunks:
            for line in chunk:
                data = line.strip().split(self.delimiter)
                var_names.append(data[0])
                var_means.append(float(data[1]))
        return np.array(var_names), np.array(var_means)

    def _get_species_scores(self, all_lines):
        """
        Read in species scores from ordination file and return species names
        as a 1 x n_species array and species centroids as a n_species x
        n_axes array

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        species_names : np.array
            Variable names associated with species

        species_scores : np.array
            Species centroids from ordination
        """

        # Get the lines associated with the species scores
        species_re = re.compile(r"^###\s+Species\s+Centroids\s+###.*")
        chunks = self.read_chunks(
            all_lines, species_re, self.blank_re, skip_lines=2, flush=True
        )

        # Extract the species names and species scores
        species_names = []
        species_scores = []
        for chunk in chunks:
            for line in chunk:
                data = line.strip().split(self.delimiter)
                species_names.append(data[0])
                species_scores.append([float(x) for x in data[1:]])
        return np.array(species_names), np.array(species_scores)

    def _get_plots(self, all_lines):
        """
        Read in plot IDs and scores from ordination file and return plot IDs as
        a 1 x n_plots array and plot scores as a n_plots x n_axes array.
        Ensure that the plot IDs and scores are sorted.

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        plot_ids : np.array
            Plot identification numbers

        plot_scores : np.array
            Plot scores within the ordination space
        """

        # Get the lines associated with the ordination variable means
        plot_re = re.compile(r"^###\s+Site\s+LC\s+Scores\s+###.*")
        chunks = self.read_chunks(
            all_lines, plot_re, self.blank_re, skip_lines=2, flush=True
        )

        # Extract the plot IDs and scores
        plot_ids = []
        plot_scores = []
        for chunk in chunks:
            for line in chunk:
                data = line.strip().split(self.delimiter)
                plot_ids.append(int(data[0]))
                plot_scores.append([float(x) for x in data[1:]])
        plot_ids = np.array(plot_ids)
        plot_scores = np.array(plot_scores)

        # Sort the plot IDs and scores and return
        ind_arr = np.argsort(plot_ids)
        plot_ids = plot_ids[ind_arr]
        plot_scores = plot_scores[ind_arr]
        return plot_ids, plot_scores

    def _get_biplot_scores(self, all_lines):
        """
        Read in variable biplot scores from ordination file and return
        variable names as a 1 x n_variables array and ordination variable
        biplot scores as a n_variables x n_axes array

        Parameters
        ----------
        all_lines : list of str
            All lines from the input ordination_file

        Returns
        -------
        var_names : np.array
            Variable names associated with the ordination variables

        biplot_scores : np.array
            Ordination variables biplot scores on each axis
        """

        # Get the lines associated with the biplot section
        biplot_re = re.compile(r"^###\s+Biplot\s+Scores\s+###.*")
        chunks = self.read_chunks(
            all_lines, biplot_re, self.blank_re, skip_lines=2, flush=True
        )

        # Extract the variable names and biplot scores
        var_names = []
        biplot_scores = []
        for chunk in chunks:
            for line in chunk:
                data = line.strip().split(self.delimiter)
                var_names.append(data[0])
                biplot_scores.append([float(x) for x in data[1:]])
        return np.array(var_names), np.array(biplot_scores)


if __name__ == "__main__":
    lop = LemmaOrdinationParser(delimiter=",")
    m = lop.parse(sys.argv[1])
    print(m)
