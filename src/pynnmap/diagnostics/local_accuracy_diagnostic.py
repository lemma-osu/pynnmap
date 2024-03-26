from collections import namedtuple

import numpy as np

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import statistics
from pynnmap.misc import utilities
from pynnmap.parser.xml_stand_metadata_parser import XMLStandMetadataParser
from pynnmap.parser.xml_stand_metadata_parser import Flags


# Namedtuple to capture the local statistics
LocalStatistics = namedtuple(
    "LocalStatistics",
    ["pearson_r", "spearman_r", "rmse", "std_rmse", "bias", "r2"],
)


class LocalAccuracyDiagnostic(diagnostic.Diagnostic):
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
        self.obs_df, self.prd_df = utilities.build_obs_prd_dataframes(
            self.observed_file, self.predicted_file, self.id_field
        )

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        p = parameter_parser
        return cls(
            p.stand_attribute_file,
            p.independent_predicted_file,
            p.stand_metadata_file,
            p.plot_id_field,
            p.local_accuracy_file,
        )

    def run_attr(self, attr):
        # Retrieve the observed and predicted values
        obs_vals = getattr(self.obs_df, attr.field_name)
        prd_vals = getattr(self.prd_df, attr.field_name)

        # Get the statistics
        rmse = statistics.rmse(obs_vals, prd_vals)
        n_rmse = rmse / obs_vals.mean() if np.any(obs_vals != 0.0) else 0.0
        return LocalStatistics(
            statistics.pearson_r(obs_vals, prd_vals),
            statistics.spearman_r(obs_vals, prd_vals),
            rmse,
            n_rmse,
            statistics.bias_percentage(obs_vals, prd_vals),
            statistics.r2(obs_vals, prd_vals),
        )

    def run_diagnostic(self):
        with open(self.statistics_file, "w") as stats_fh:
            out_list = [
                "VARIABLE",
                "PEARSON_R",
                "SPEARMAN_R",
                "RMSE",
                "NORMALIZED_RMSE",
                "BIAS_PERCENTAGE",
                "R_SQUARE",
            ]
            stats_fh.write(",".join(out_list) + "\n")

            # Read in the stand attribute metadata and get continuous attributes
            mp = XMLStandMetadataParser(self.stand_metadata_file)
            attrs = mp.filter(Flags.CONTINUOUS | Flags.ACCURACY)

            # For each attribute, calculate the statistics
            for attr in attrs:
                # Skip any missing attributes
                if attr.field_name not in self.obs_df.columns:
                    print(f"{attr.field_name} is not present in observed file")
                    continue

                stats = self.run_attr(attr)

                # Print this out to the stats file
                out_list = [
                    attr.field_name,
                    "{:.6f}".format(stats.pearson_r),
                    "{:.6f}".format(stats.spearman_r),
                    "{:.6f}".format(stats.rmse),
                    "{:.6f}".format(stats.std_rmse),
                    "{:.6f}".format(stats.bias),
                    "{:.6f}".format(stats.r2),
                ]
                stats_fh.write(",".join(out_list) + "\n")
