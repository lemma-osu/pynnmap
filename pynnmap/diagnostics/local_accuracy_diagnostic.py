import numpy as np
import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import statistics
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class LocalAccuracyDiagnostic(diagnostic.Diagnostic):
    _required = ['observed_file', 'predicted_file', 'stand_metadata_file']

    def __init__(
            self, observed_file, predicted_file, stand_metadata_file,
            id_field, statistics_file):
        self.observed_file = observed_file
        self.predicted_file = predicted_file
        self.stand_metadata_file = stand_metadata_file
        self.id_field = id_field
        self.statistics_file = statistics_file

        self.check_missing_files()

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        p = parameter_parser
        return cls(
            p.stand_attribute_file,
            p.independent_predicted_file,
            p.stand_metadata_file,
            p.plot_id_field,
            p.local_accuracy_file
        )

    def run_diagnostic(self):

        # Open the stats file and print out the header line
        stats_fh = open(self.statistics_file, 'w')
        out_list = [
            'VARIABLE',
            'PEARSON_R',
            'SPEARMAN_R',
            'RMSE',
            'NORMALIZED_RMSE',
            'BIAS_PERCENTAGE',
            'R_SQUARE',
        ]
        stats_fh.write(','.join(out_list) + '\n')

        # Read the observed and predicted files into dataframes
        obs = pd.read_csv(self.observed_file, low_memory=False)
        prd = pd.read_csv(self.predicted_file, low_memory=False)

        # Subset the observed data just to the IDs that are in the
        # predicted file
        obs_keep = np.in1d(
            getattr(obs, self.id_field), getattr(prd, self.id_field))
        obs = obs[obs_keep]

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # For each variable, calculate the statistics
        for v in obs.columns:

            # Get the metadata for this field
            try:
                fm = mp.get_attribute(v)
            except ValueError:
                err_msg = 'Missing metadata for {}'.format(v)
                # TODO: log this as warning instead
                print(err_msg)
                continue

            # Only continue if this is a continuous accuracy variable
            if not fm.is_continuous_accuracy_attr():
                continue

            obs_vals = getattr(obs, v)
            prd_vals = getattr(prd, v)

            if np.all(obs_vals == 0.0):
                pearson_r = 0.0
                spearman_r = 0.0
                rmse = 0.0
                std_rmse = 0.0
                bias = 0.0
                r2 = 0.0
            else:
                if np.all(prd_vals == 0.0):
                    pearson_r = 0.0
                    spearman_r = 0.0
                else:
                    pearson_r = statistics.pearson_r(obs_vals, prd_vals)
                    spearman_r = statistics.spearman_r(obs_vals, prd_vals)
                rmse = statistics.rmse(obs_vals, prd_vals)
                std_rmse = rmse / obs_vals.mean()
                bias = statistics.bias_percentage(obs_vals, prd_vals)
                r2 = statistics.r2(obs_vals, prd_vals)

            # Print this out to the stats file
            out_list = [
                v,
                '%.6f' % pearson_r,
                '%.6f' % spearman_r,
                '%.6f' % rmse,
                '%.6f' % std_rmse,
                '%.6f' % bias,
                '%.6f' % r2,
            ]
            stats_fh.write(','.join(out_list) + '\n')
        stats_fh.close()
