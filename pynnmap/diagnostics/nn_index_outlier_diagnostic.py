import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.misc.utilities import df_to_csv


class NNIndexOutlierDiagnostic(diagnostic.Diagnostic):
    _required = ['nn_index_file']

    def __init__(self, parameters):
        self.nn_index_file = parameters.dependent_nn_index_file
        self.nn_index_outlier_file = parameters.nn_index_outlier_file
        self.index_threshold = parameters.index_threshold
        self.id_field = parameters.plot_id_field

        self.check_missing_files()

    def run_diagnostic(self):
        # Read in the dependent nn_index_file
        in_df = pd.read_csv(self.nn_index_file)

        # Subset the observed data to just those values above the
        # index threshold
        in_df = in_df[in_df.AVERAGE_POSITION >= self.index_threshold]

        # Write out the resulting recarray
        df_to_csv(in_df, self.nn_index_outlier_file)
