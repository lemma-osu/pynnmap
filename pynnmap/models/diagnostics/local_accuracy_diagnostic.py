import numpy as np

from models.diagnostics import diagnostic
from models.misc import statistics
from models.misc import utilities
from models.parser import parameter_parser as pp
from models.parser import xml_stand_metadata_parser as xsmp


class LocalAccuracyDiagnostic(diagnostic.Diagnostic):

    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):
                self.observed_file = p.stand_attribute_file
                self.predicted_file = p.independent_predicted_file
                self.stand_metadata_file = p.stand_metadata_file
                self.statistics_file = p.local_accuracy_file
                self.id_field = 'FCID'
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            try:
                self.observed_file = kwargs['observed_file']
                self.predicted_file = kwargs['independent_predicted_file']
                self.stand_metadata_file = kwargs['stand_metadata_file']
                self.statistics_file = kwargs['local_accuracy_file']
                self.id_field = kwargs['id_field']
            except KeyError:
                err_msg = 'Not all required parameters were passed'
                raise ValueError(err_msg)

        # Ensure all input files are present
        files = [self.observed_file, self.predicted_file,
            self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping LocalAccuracyDiagnostic\n'
            raise e

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

        # Read the observed and predicted files into numpy recarrays
        obs = utilities.csv2rec(self.observed_file)
        prd = utilities.csv2rec(self.predicted_file)

        # Subset the observed data just to the IDs that are in the
        # predicted file
        obs_keep = np.in1d(
            getattr(obs, self.id_field), getattr(prd, self.id_field))
        obs = obs[obs_keep]

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # For each variable, calculate the statistics
        for v in obs.dtype.names:

            # Get the metadata for this field
            try:
                fm = mp.get_attribute(v)
            except:
                err_msg = v + ' is missing metadata.'
                print err_msg
                continue

            # Only continue if this is a continuous accuracy variable
            if fm.field_type != 'CONTINUOUS' or fm.accuracy_attr == 0:
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
