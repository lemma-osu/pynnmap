import numpy as np
from models.diagnostics import diagnostic
from models.misc import statistics
from models.misc import utilities
from models.parser import parameter_parser as pp
from models.parser import xml_stand_metadata_parser as xsmp


class SpeciesAccuracyDiagnostic(diagnostic.Diagnostic):
    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):
                self.observed_file = p.stand_attribute_file
                self.predicted_file = p.independent_predicted_file
                self.stand_metadata_file = p.stand_metadata_file
                self.statistics_file = p.species_accuracy_file
                self.id_field = 'FCID'
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            err_msg = 'Only ParameterParser objects may be passed.'
            raise NotImplementedError(err_msg)

        # Ensure all input files are present
        files = [self.observed_file, self.predicted_file,
            self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping SpeciesAccuracyDiagnostic\n'
            raise e

    def run_diagnostic(self):
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

        # Open the stats file and print out the header lines
        stats_fh = open(self.statistics_file, 'w')
        out_list = [
            'SPECIES',
            'OP_PP',
            'OP_PA',
            'OA_PP',
            'OA_PA',
            'PREVALENCE',
            'SENSITIVITY',
            'FALSE_NEGATIVE_RATE',
            'SPECIFICITY',
            'FALSE_POSITIVE_RATE',
            'PERCENT_CORRECT',
            'ODDS_RATIO',
            'KAPPA',
        ]
        stats_fh.write(','.join(out_list) + '\n')

        # For each variable, calculate the statistics
        for v in obs.dtype.names:

            # Get the metadata for this field
            try:
                fm = mp.get_attribute(v)
            except:
                err_msg = v + ' is missing metadata.'
                print err_msg
                continue

            # Only continue if this is a continuous species variable
            if fm.field_type != 'CONTINUOUS' or fm.species_attr == 0:
                continue

            obs_vals = getattr(obs, v)
            prd_vals = getattr(prd, v)

            # Create a binary error matrix from the obs and prd data
            stats = statistics.BinaryErrorMatrix(obs_vals, prd_vals)
            counts = stats.counts()

            # Build the list of items for printing
            out_list = [
                v,
                '%d' % counts[0, 0],
                '%d' % counts[0, 1],
                '%d' % counts[1, 0],
                '%d' % counts[1, 1],
                '%.4f' % stats.prevalence(),
                '%.4f' % stats.sensitivity(),
                '%.4f' % stats.false_negative_rate(),
                '%.4f' % stats.specificity(),
                '%.4f' % stats.false_positive_rate(),
                '%.4f' % stats.percent_correct(),
                '%.4f' % stats.odds_ratio(),
                '%.4f' % stats.kappa(),
            ]
            stats_fh.write(','.join(out_list) + '\n')

        stats_fh.close()
