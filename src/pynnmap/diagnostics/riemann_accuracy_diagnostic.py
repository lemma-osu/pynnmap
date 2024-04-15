from __future__ import annotations

import os
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd

from ..core import get_id_list, get_independence_filter
from ..core.attribute_predictor import ContinuousAttributePredictor
from ..core.nn_finder import PixelNNFinder
from ..core.prediction_output import subset_neighbors
from ..core.stand_attributes import StandAttributes
from ..misc.utilities import df_to_csv
from ..parser import xml_stand_metadata_parser as xsmp
from ..parser.xml_stand_metadata_parser import Flags
from . import diagnostic

RiemannComparison = namedtuple(
    "RiemannComparison", ["prefix", "obs_file", "prd_file", "id_field", "k"]
)


class ECDF:
    def __init__(self, observations):
        self.observations = observations

    def __call__(self, x):
        try:
            return (self.observations <= x).mean()
        except AttributeError:
            self.observations = np.array(self.observations)
            return (self.observations <= x).mean()


class RiemannVariable:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def gmfr_statistics(self):
        # Short circuit the condition where there is no observed or predicted
        # variance typically caused by no observed or predicted presences
        if self.x.var() == 0.0 or self.y.var() == 0.0:
            return {
                "gmfr_a": 0.0,
                "gmfr_b": 0.0,
                "ac": 0.0,
                "ac_sys": 0.0,
                "ac_uns": 0.0,
            }

        x_mean = self.x.mean()
        y_mean = self.y.mean()

        b = np.sqrt(self.y.var() / self.x.var())
        a = y_mean - (b * x_mean)

        # Sum of squared differences
        ssd = ((self.x - self.y) * (self.x - self.y)).sum()

        # Sum of product differences for GMFR for systematic and unsystematic
        # differences
        c = -a / b
        d = 1.0 / b
        y_hat = a + b * self.x
        x_hat = c + d * self.y
        spd_u = (np.abs(self.x - x_hat) * np.abs(self.y - y_hat)).sum()
        spd_s = ssd - spd_u

        # Sum of potential differences
        term1 = np.abs(x_mean - y_mean) + np.abs(self.x - x_mean)
        term2 = np.abs(x_mean - y_mean) + np.abs(self.y - y_mean)
        spod = (term1 * term2).sum()

        # Agreement coefficients for total, systematic and
        # unsystematic differences
        ac = 1.0 - (ssd / spod)
        ac_sys = 1.0 - (spd_s / spod)
        ac_uns = 1.0 - (spd_u / spod)

        return {
            "gmfr_a": a,
            "gmfr_b": b,
            "ac": ac,
            "ac_sys": ac_sys,
            "ac_uns": ac_uns,
        }

    def ks_statistics(self, num_bins=1000):
        x_sorted = np.sort(self.x)
        y_sorted = np.sort(self.y)

        global_max = np.max(np.hstack((x_sorted, y_sorted)))
        global_min = np.min(np.hstack((x_sorted, y_sorted)))

        bins = np.linspace(global_min, global_max, num_bins)

        x_ecdf = ECDF(x_sorted)
        y_ecdf = ECDF(y_sorted)

        x_freq = np.array([x_ecdf(i) for i in bins])
        y_freq = np.array([y_ecdf(i) for i in bins])
        diff_freq = np.abs(x_freq - y_freq)
        ks_max = np.max(diff_freq)
        ks_mean = np.mean(diff_freq)

        return {"ks_max": ks_max, "ks_mean": ks_mean}


class RiemannAccuracyDiagnostic(diagnostic.Diagnostic):
    _required: list[str] = ["hex_attribute_file", "hex_id_file"]

    def __init__(self, parameter_parser):
        self.parameter_parser = p = parameter_parser
        self.id_field = p.plot_id_field

        # We want to check on the existence of the hex attribute
        # file before running, so make this an instance property
        self.hex_attribute_file = p.hex_attribute_file
        self.hex_id_file = p.hex_id_file

        self.check_missing_files()

        # Read in the hex ID crosswalk file
        self.hex_id_xwalk = pd.read_csv(self.hex_id_file)

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        return cls(parameter_parser)

    @staticmethod
    def _create_directory_structure(hex_resolutions, root_dir):
        hex_levels = [f"hex_{x[1]}" for x in hex_resolutions]
        all_levels = ["plot_pixel"] + hex_levels
        for level in all_levels:
            sub_dir = os.path.join(root_dir, level)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

    def run_diagnostic(self):
        # Shortcut to the parameter parser and set up often used fields
        p = self.parameter_parser
        id_field = p.plot_id_field
        root_dir = p.riemann_output_folder
        k_values = p.riemann_k_values
        hex_resolutions = p.riemann_hex_resolutions

        # Create the directory structure based on the hex levels
        self._create_directory_structure(hex_resolutions, root_dir)

        # Get the stand attributes and filter to continuous accuracy fields
        attr_fn = self.hex_attribute_file
        mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=id_field)
        flags = Flags.CONTINUOUS | Flags.ACCURACY
        attrs = list(attr_data.get_attr_df(flags=flags).columns)

        # Write out the plot_pixel observed file
        file_name = "plot_pixel_observed.csv"
        output_file = os.path.join(root_dir, "plot_pixel", file_name)
        plot_pixel_obs = attr_data.get_attr_df(flags=flags).astype(np.float64)
        df_to_csv(plot_pixel_obs, output_file, index=True)

        # Get the IDs on which to run AA
        ids = get_id_list(self.hex_attribute_file, self.id_field)

        # Create a PixelNNFinder object and calculate neighbors and distances
        finder = PixelNNFinder(p)
        neighbor_data = finder.calculate_neighbors_at_ids(ids)

        # Create an independence filter based on the relationship of the
        # id_field and the no_self_assign_field
        fltr = get_independence_filter(p)

        # Create a plot attribute predictor instance
        model_attr_fn = p.stand_attribute_file
        model_attr_data = StandAttributes(model_attr_fn, mp, id_field=id_field)
        plot_attr_predictor = ContinuousAttributePredictor(model_attr_data, fltr)

        # Iterate over values of k to calculate plot-pixel values
        for k, w in k_values:
            # Set weights correctly
            # TODO: Duplicate code with PredictionOutput.get_weights()
            if w is not None:
                if len(w) != k:
                    raise ValueError("Length of weights does not equal k")
                w = np.array(w).reshape(1, len(w)).T

            # Construct the output file name
            file_name = f"plot_pixel_predicted_k{k}.csv"
            output_file = os.path.join(root_dir, "plot_pixel", file_name)

            # Subset neighbors to just this k
            plot_predictions = subset_neighbors(neighbor_data, k, fltr)

            # Calculate the predictions
            predictions = plot_attr_predictor.calculate_predictions(
                plot_predictions, k=k, weights=w
            )

            # Get the predicted attributes
            df = plot_attr_predictor.get_predicted_attributes_df(
                predictions, self.id_field
            )

            # Subset columns down to just columns present in the hex
            # attribute file and write out
            df_to_csv(df[attrs].copy(), output_file, index=True)

        # Create the fields for which to extract statistics at the hexagon
        # levels
        mean_list = [(id_field, len), *[(x, np.mean) for x in attrs]]
        mean_dict = OrderedDict(mean_list)

        sd_list = [(id_field, len), *[(x, lambda i: np.std(i)) for x in attrs]]
        sd_dict = OrderedDict(sd_list)

        stat_sets = {
            "mean": mean_dict,
            "std": sd_dict,
        }

        # For each hexagon level, associate the plots with their hexagon ID
        # and find observed and predicted statistics for each hexagon
        for hex_resolution in hex_resolutions:
            hex_id_field, hex_distance = hex_resolution[:2]
            min_plots_per_hex = hex_resolution[3]
            prefix = f"hex_{hex_distance}"

            # Create a crosswalk between the id_field and the hex_id_field
            s1 = self.hex_id_xwalk[id_field]
            s2 = self.hex_id_xwalk[hex_id_field]
            id_x_hex = dict(zip(s1, s2))

            # Iterate over all sets of statistics and write a unique file
            # for each set
            for stat_name, stat_fields in stat_sets.items():
                # Get the output file name
                file_name = f"{prefix}_observed_{stat_name}.csv"
                obs_out_file = os.path.join(root_dir, prefix, file_name)

                s = pd.Series(id_x_hex, name=hex_id_field)
                df = plot_pixel_obs.merge(s, left_index=True, right_index=True)
                df[id_field] = df.index
                grouped = df.groupby(hex_id_field)
                agg_df = grouped.agg(stat_fields).rename(
                    columns={id_field: "PLOT_COUNT"}
                )
                agg_df = agg_df[min_plots_per_hex <= agg_df.PLOT_COUNT]
                df_to_csv(agg_df, obs_out_file, index=True)

            # Iterate over values of k for the predicted values
            for k, _ in k_values:
                # Open the plot_pixel predicted file for this value of k
                # and join the hex_id_field to the recarray
                prd_file = f"plot_pixel_predicted_k{k}.csv"
                prd_file = os.path.join(root_dir, "plot_pixel", prd_file)
                prd_data = pd.read_csv(prd_file)
                prd_data = prd_data.merge(self.hex_id_xwalk, on=id_field)

                # Iterate over all sets of statistics and write a unique file
                # for each set
                for stat_name, stat_fields in stat_sets.items():
                    # Get the output file name
                    file_name = f"{prefix}_predicted_k{k}_{stat_name}.csv"
                    prd_out_file = os.path.join(root_dir, prefix, file_name)

                    grouped = prd_data.groupby(hex_id_field)
                    agg_df = grouped.agg(stat_fields).rename(
                        columns={id_field: "PLOT_COUNT"}
                    )
                    agg_df = agg_df[min_plots_per_hex <= agg_df.PLOT_COUNT]
                    df_to_csv(agg_df, prd_out_file, index=True)

        # Calculate the ECDF and AC statistics
        # For ECDF and AC, it is a paired comparison between the observed
        # and predicted data.  We do this at each value of k and for each
        # hex resolution level.

        # Open the stats file
        stats_file = p.hex_statistics_file
        stats_fh = open(stats_file, "w")
        header_fields = ["LEVEL", "K", "VARIABLE", "STATISTIC", "VALUE"]
        stats_fh.write(",".join(header_fields) + "\n")

        # Create a list of RiemannComparison instances which store the
        # information needed to do comparisons between observed and predicted
        # files for any level or value of k
        compare_list = []
        for hex_resolution in hex_resolutions:
            (hex_id_field, hex_distance) = hex_resolution[:2]
            prefix = f"hex_{hex_distance}"
            obs_file = f"{prefix}_observed_mean.csv"
            obs_file = os.path.join(root_dir, prefix, obs_file)
            for k, _ in k_values:
                prd_file = f"{prefix}_predicted_k{k}_mean.csv"
                prd_file = os.path.join(root_dir, prefix, prd_file)
                r = RiemannComparison(prefix, obs_file, prd_file, hex_id_field, k)
                compare_list.append(r)

        # Add the plot_pixel comparisons to this list
        prefix = "plot_pixel"
        obs_file = "plot_pixel_observed.csv"
        obs_file = os.path.join(root_dir, prefix, obs_file)
        for k, _ in k_values:
            prd_file = f"plot_pixel_predicted_k{k}.csv"
            prd_file = os.path.join(root_dir, prefix, prd_file)
            r = RiemannComparison(prefix, obs_file, prd_file, id_field, k)
            compare_list.append(r)

        # Do all the comparisons
        for c in compare_list:
            obs_data = pd.read_csv(c.obs_file)
            prd_data = pd.read_csv(c.prd_file)

            # Ensure that the IDs between the observed and predicted
            # data line up
            ids1 = getattr(obs_data, c.id_field)
            ids2 = getattr(prd_data, c.id_field)
            if np.all(ids1 != ids2):
                err_msg = "IDs do not match between observed and predicted data"
                raise ValueError(err_msg)

            for attr in attrs:
                arr1 = getattr(obs_data, attr)
                arr2 = getattr(prd_data, attr)
                rv = RiemannVariable(arr1, arr2)

                gmfr_stats = rv.gmfr_statistics()
                for stat in ("gmfr_a", "gmfr_b", "ac", "ac_sys", "ac_uns"):
                    stat_line = "%s,%d,%s,%s,%.4f\n" % (
                        c.prefix.upper(),
                        c.k,
                        attr,
                        stat.upper(),
                        gmfr_stats[stat],
                    )
                    stats_fh.write(stat_line)

                ks_stats = rv.ks_statistics()
                for stat in ("ks_max", "ks_mean"):
                    stat_line = "%s,%d,%s,%s,%.4f\n" % (
                        c.prefix.upper(),
                        c.k,
                        attr,
                        stat.upper(),
                        ks_stats[stat],
                    )
                    stats_fh.write(stat_line)
