import numpy as np
import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import statistics
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class SpeciesAccuracyDiagnostic(diagnostic.Diagnostic):
    _required = ["observed_file", "predicted_file", "stand_metadata_file"]

    def __init__(
        self,
        observed_file,
        predicted_file,
        stand_metadata_file,
        id_field,
        statistics_file,
    ):
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
            p.species_accuracy_file,
        )

    def run_diagnostic(self):
        # Read the observed and predicted files into numpy recarrays
        obs = pd.read_csv(self.observed_file, low_memory=False)
        prd = pd.read_csv(self.predicted_file, low_memory=False)

        # Subset the observed data just to the IDs that are in the
        # predicted file
        obs_keep = np.in1d(
            getattr(obs, self.id_field), getattr(prd, self.id_field)
        )
        obs = obs[obs_keep]

        # Read in the stand attribute metadata
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        with open(self.statistics_file, "w") as stats_fh:
            out_list = [
                "SPECIES",
                "OP_PP",
                "OP_PA",
                "OA_PP",
                "OA_PA",
                "PREVALENCE",
                "SENSITIVITY",
                "FALSE_NEGATIVE_RATE",
                "SPECIFICITY",
                "FALSE_POSITIVE_RATE",
                "PERCENT_CORRECT",
                "ODDS_RATIO",
                "KAPPA",
            ]
            stats_fh.write(",".join(out_list) + "\n")

                # For each variable, calculate the statistics
            for v in obs.columns:

                        # Get the metadata for this field
                try:
                    fm = mp.get_attribute(v)
                except ValueError:
                    err_msg = f"Missing metadata for {v}"
                    # TODO: log this as warning instead
                    print(err_msg)
                    continue

                # Only continue if this is a continuous species variable
                if not fm.is_continuous_species_attr():
                    continue

                obs_vals = getattr(obs, v)
                prd_vals = getattr(prd, v)

                # Create a binary error matrix from the obs and prd data
                stats = statistics.BinaryErrorMatrix(obs_vals, prd_vals)
                counts = stats.counts()

                # Build the list of items for printing
                out_list = [
                    v,
                    "%d" % counts[0, 0],
                    "%d" % counts[0, 1],
                    "%d" % counts[1, 0],
                    "%d" % counts[1, 1],
                    "%.4f" % stats.prevalence(),
                    "%.4f" % stats.sensitivity(),
                    "%.4f" % stats.false_negative_rate(),
                    "%.4f" % stats.specificity(),
                    "%.4f" % stats.false_positive_rate(),
                    "%.4f" % stats.percent_correct(),
                    "%.4f" % stats.odds_ratio(),
                    "%.4f" % stats.kappa(),
                ]
                stats_fh.write(",".join(out_list) + "\n")
