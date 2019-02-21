import numpy as np
import pandas as pd

from pynnmap.diagnostics import diagnostic


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
        # Run this for both independent and dependent predictions
        out_dfs = []
        for (prd_type, prd_file) in self.predicted_files:
            # Read the observed and predicted files into data frames
            obs_df = pd.read_csv(self.observed_file, index_col=self.id_field)
            prd_df = pd.read_csv(prd_file, index_col=self.id_field)

            # Subset the observed data just to the IDs that are in the
            # predicted file
            obs_df = obs_df[obs_df.index.isin(prd_df.index)]

            # Iterate over the list of deviation variables, capturing the plots
            # that exceed the minimum threshold specified
            columns = [
                self.id_field, 'PREDICTION_TYPE', 'VARIABLE', 'OBSERVED_VALUE',
                'PREDICTED_VALUE', 'DEVIATION'
            ]
            for (variable, min_deviation) in self.deviation_variables:
                df = pd.DataFrame({
                    self.id_field: obs_df.index,
                    'PREDICTION_TYPE': prd_type.upper(),
                    'VARIABLE': variable,
                    'OBSERVED_VALUE': obs_df[variable],
                    'PREDICTED_VALUE': prd_df[variable],
                    'DEVIATION': obs_df[variable] - prd_df[variable]
                }, columns=columns)

                # Subset to just those deviations over the min_deviation
                df = df[np.abs(df.DEVIATION) >= min_deviation]
                if len(df):
                    out_dfs.append(df)

        # Create a master dataframe of all outliers
        all_df = pd.concat(out_dfs)

        # Write this out
        all_df.to_csv(self.vd_output_file, index=False, float_format='%.4f')
