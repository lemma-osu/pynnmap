from __future__ import annotations

import pandas as pd

from ..misc.utilities import df_to_csv
from . import diagnostic


class NNIndexOutlierDiagnostic(diagnostic.Diagnostic):
    _required: list[str] = ["nn_index_file"]

    def __init__(self, parameters):
        self.nn_index_file = parameters.dependent_nn_index_file
        self.outlier_filename = parameters.nn_index_outlier_file
        self.index_threshold = parameters.index_threshold
        self.id_field = parameters.plot_id_field

        self.check_missing_files()

    def run_diagnostic(self):
        # Read in the dependent nn_index_file
        in_df = pd.read_csv(self.nn_index_file)

        # Subset the observed data to just those values above the
        # index threshold
        in_df = in_df[self.index_threshold <= in_df.AVERAGE_POSITION]

        # Write out the resulting recarray
        df_to_csv(in_df, self.outlier_filename)
