import numpy as np

from models.diagnostics import diagnostic
from models.diagnostics import vegetation_class_diagnostic as vcd
from models.misc import utilities


class VegetationClassOutlierDiagnostic(vcd.VegetationClassDiagnostic):

    def __init__(self, parameters):
        self.observed_file = parameters.stand_attribute_file
        self.vegclass_outlier_file = parameters.vegclass_outlier_file
        self.id_field = parameters.summary_level + 'ID'

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

        # Open the outlier file and write the header line
        vc_outlier_fh = open(self.vegclass_outlier_file, 'w')
        header_fields = (
            self.id_field,
            'PREDICTION_TYPE',
            'OBSERVED_VEGCLASS',
            'PREDICTED_VEGCLASS',
            'OUTLIER_TYPE'
        )
        vc_outlier_fh.write(','.join(header_fields) + '\n')

        # Run this for both independent and dependent predictions
        for (prd_type, prd_file) in self.predicted_files:

            # Read the observed and predicted files into numpy recarrays
            obs = utilities.csv2rec(self.observed_file)
            prd = utilities.csv2rec(prd_file)

            # Subset the observed data just to the IDs that are in the
            # predicted file
            obs_keep = np.in1d(
                getattr(obs, self.id_field), getattr(prd, self.id_field))
            obs = obs[obs_keep]

            # Calculate VEGCLASS for both the observed and predicted data
            vc_dict = self.vegclass_aa(obs, prd, id_field=self.id_field)

            # Find the outliers
            outliers = self.find_vegclass_outliers(vc_dict)

            # Print out the outliers
            for outlier in outliers:
                (id, obs_vc, prd_vc, outlier_type) = outlier
                out_fields = (
                    '%d' % id,
                    '%s' % prd_type.upper(),
                    '%d' % obs_vc,
                    '%d' % prd_vc,
                    '%s' % outlier_type,
                )
                vc_outlier_fh.write(','.join(out_fields) + '\n')

        vc_outlier_fh.close()

    def find_vegclass_outliers(self, vc_dict):
        red_outliers = {
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

        orange_outliers = {
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

        yellow_outliers = {
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

        outliers = []
        for id in sorted(vc_dict):
            observed = vc_dict[id]['obs_vc']
            predicted = vc_dict[id]['prd_vc']
            if predicted in yellow_outliers[observed]:
                rec = (id, observed, predicted, 'yellow')
                outliers.append(rec)
            if predicted in orange_outliers[observed]:
                rec = (id, observed, predicted, 'orange')
                outliers.append(rec)
            if predicted in red_outliers[observed]:
                rec = (id, observed, predicted, 'red')
                outliers.append(rec)
        return outliers

    def get_outlier_filename(self):
        return self.vegclass_outlier_file
