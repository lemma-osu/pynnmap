import numpy as np
import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.diagnostics import vegetation_class_diagnostic as vcd

# Define the classes of vegetation class outliers.  Red outliers represent
# large differences between observed and predicted vegetation classes, with
# orange and yellow being less severe
RED_OUTLIERS = {
    1: [10, 11],
    2: [11],
    3: [],
    4: [],
    5: [11],
    6: [],
    7: [],
    8: [11],
    9: [],
    10: [1],
    11: [1, 2, 5, 8],
}

ORANGE_OUTLIERS = {
    1: [4, 6, 7, 8, 9],
    2: [10],
    3: [10, 11],
    4: [1],
    5: [10],
    6: [1, 11],
    7: [1],
    8: [1],
    9: [1],
    10: [2, 3, 5],
    11: [3, 6],
}

YELLOW_OUTLIERS = {
    1: [3, 5],
    2: [4, 6, 7, 9],
    3: [1, 6, 7, 8, 9],
    4: [2, 5, 9, 10, 11],
    5: [1, 4, 7, 9],
    6: [2, 3, 8, 10],
    7: [2, 3, 5, 8, 9],
    8: [3, 4, 6, 7, 10],
    9: [2, 3, 4, 5, 7, 11],
    10: [4, 6, 8],
    11: [4, 9],
}


def find_vegclass_outlier_class(rec):
    observed, predicted = rec['OBSERVED'], rec['PREDICTED']
    if predicted in YELLOW_OUTLIERS[observed]:
        return 'yellow'
    elif predicted in ORANGE_OUTLIERS[observed]:
        return 'orange'
    elif predicted in RED_OUTLIERS[observed]:
        return 'red'
    return 'green'


class VegetationClassOutlierDiagnostic(vcd.VegetationClassDiagnostic):

    def __init__(self, parameters):
        self.observed_file = parameters.stand_attribute_file
        self.vegclass_outlier_file = parameters.vegclass_outlier_file
        self.id_field = parameters.plot_id_field

        # Create a list of predicted files - both independent and dependent
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
            e.message += '\nSkipping VegetationClassOutlierDiagnostic\n'
            raise e

    def run_diagnostic(self):
        # Run this for both independent and dependent predictions
        out_dfs = []
        for (prd_type, prd_file) in self.predicted_files:

            # Read the observed and predicted files into dataframes
            obs_df = pd.read_csv(self.observed_file, index_col=self.id_field)
            prd_df = pd.read_csv(prd_file, index_col=self.id_field)

            # Subset the observed data just to the IDs that are in the
            # predicted file
            obs_df = obs_df[obs_df.index.isin(prd_df.index)]
            obs_df.reset_index(inplace=True)
            prd_df.reset_index(inplace=True)

            # Calculate VEGCLASS for both the observed and predicted data
            vc_df = self.vegclass_aa(obs_df, prd_df, id_field=self.id_field)
            vc_df.columns = [self.id_field, 'OBSERVED', 'PREDICTED']

            # Find the outliers
            vc_df['CLASS'] = vc_df.apply(find_vegclass_outlier_class, axis=1)

            # Only keep yellow, orange, and red outliers
            vc_df = vc_df[vc_df.CLASS != 'green']

            # Format this dataframe for export and append it to the out_df list
            vc_df.insert(1, 'PREDICTION_TYPE', prd_type.upper())
            vc_df.rename(columns={
                'OBSERVED': 'OBSERVED_VEGCLASS',
                'PREDICTED': 'PREDICTED_VEGCLASS',
                'CLASS': 'OUTLIER_TYPE'
            }, inplace=True)
            out_dfs.append(vc_df)

        # Merge together the dfs and export
        out_df = pd.concat(out_dfs)
        out_df.to_csv(self.vegclass_outlier_file, index=False)

    def get_outlier_filename(self):
        return self.vegclass_outlier_file
