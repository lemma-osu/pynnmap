import pandas as pd

from pynnmap.diagnostics import diagnostic


class NNIndexOutlierDiagnostic(diagnostic.Diagnostic):

    def __init__(self, parameters):
        self.nn_index_file = parameters.dependent_nn_index_file
        self.nn_index_outlier_file = parameters.nn_index_outlier_file
        self.index_threshold = parameters.index_threshold
        self.id_field = parameters.plot_id_field

        # Ensure all input files are present
        files = [
            self.nn_index_file,
        ]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping NNIndexOutlierDiagnostic\n'
            raise e

    def run_diagnostic(self):
        # Read in the dependent nn_index_file
        in_data = pd.read_csv(self.nn_index_file)

        # Subset the observed data to just those values above the
        # index threshold
        in_data = in_data[in_data.AVERAGE_POSITION >= self.index_threshold]

        # Write out the resulting recarray
        in_data.to_csv(
            self.nn_index_outlier_file, index=False, float_format='%.4f')

    def get_outlier_filename(self):
        return self.nn_index_outlier_file
