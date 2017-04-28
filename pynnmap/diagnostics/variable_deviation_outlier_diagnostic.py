import numpy as np

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import utilities


class VariableDeviationOutlierDiagnostic(diagnostic.Diagnostic):

    def __init__(self, parameters):
        self.observed_file = parameters.stand_attribute_file
        self.vd_output_file = parameters.variable_deviation_file
        self.id_field = parameters.plot_id_field
        self.deviation_variables = parameters.deviation_variables

        # Create a list of prediction files - both independent and dependent
        self.predicted_files = [
            ('dependent', parameters.dependent_predicted_file),
            ('independent', parameters.independent_predicted_file),
        ]

        # Ensure all input files are present
        files = [
            self.observed_file,
            parameters.dependent_predicted_file,
            parameters.independent_predicted_file,
        ]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping VariableDeviationOutlierDiagnostic\n'
            raise e

    def run_diagnostic(self):
        # Open the output file and write the header
        out_fh = open(self.vd_output_file, 'w')
        out_fh.write(
            '%s,PREDICTION_TYPE,VARIABLE,OBSERVED_VALUE,PREDICTED_VALUE\n' % (
                self.id_field))

        # Run this for both independent and dependent predictions
        for (prd_type, prd_file) in self.predicted_files:

            # Read the observed and predicted files into numpy recarrays
            obs_data = utilities.csv2rec(self.observed_file)
            prd_data = utilities.csv2rec(prd_file)

            # Subset the observed data just to the IDs that are in the
            # predicted file
            obs_keep = np.in1d(
                getattr(obs_data, self.id_field),
                getattr(prd_data, self.id_field))
            obs_data = obs_data[obs_keep]

            # Iterate over the list of deviation variables, capturing the plots
            # that exceed the minimum threshold specified
            outliers = {}
            for (variable, min_deviation) in self.deviation_variables:
                obs_vals = getattr(obs_data, variable)
                prd_vals = getattr(prd_data, variable)
                abs_diff_vals = np.abs(obs_vals - prd_vals)
                indexes = np.argwhere(abs_diff_vals >= min_deviation)
                outliers[variable] = indexes

            # Create the file of outliers
            for (variable, min_deviation) in self.deviation_variables:
                outlier_list = outliers[variable]
                for index in outlier_list:
                    obs_row = obs_data[index]
                    prd_row = prd_data[index]
                    id = getattr(obs_row, self.id_field)
                    obs_val = getattr(obs_row, variable)
                    prd_val = getattr(prd_row, variable)
                    diff_val = obs_val - prd_val
                    out_data = [
                        '%d' % id,
                        '%s' % prd_type.upper(),
                        '%s' % variable,
                        '%.4f' % obs_val,
                        '%.4f' % prd_val,
                        '%.4f' % diff_val,
                    ]
                    out_fh.write(','.join(out_data) + '\n')

        # Clean up
        out_fh.close()
