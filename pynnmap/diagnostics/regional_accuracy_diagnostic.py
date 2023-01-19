import os

import numpy as np
import pandas as pd
import rasterio

from pynnmap.cli import build_attribute_raster
from pynnmap.core import get_id_list
from pynnmap.core.nn_finder import PixelNNFinder
from pynnmap.core.prediction_output import IndependentOutput
from pynnmap.diagnostics import diagnostic
from pynnmap.diagnostics.error_matrix_diagnostic import ErrorMatrixDiagnostic
from pynnmap.diagnostics.olofsson_diagnostic import OlofssonDiagnostic
from pynnmap.misc import histogram
from pynnmap.misc import interval_classification as ic
from pynnmap.misc.weighted_array import WeightedArray
from pynnmap.parser.xml_stand_metadata_parser import XMLStandMetadataParser
from pynnmap.parser.xml_stand_metadata_parser import Flags


def get_predicted_raster(attr):
    # TODO: This is *very* fragile
    fn = f"./attribute_rasters/{attr.field_name.lower()}.tif"
    with rasterio.open(fn) as src:
        arr = src.read(1, masked=True)
        cell_area = src.res[0] * src.res[1]
        scalar = float(src.tags().get("SCALAR") or 1.0)

    # Get the count of the nonforest pixels
    nf_count = np.ma.where(arr < 0, 1, 0).sum()
    hectares_per_pixel = cell_area / 10000.0
    predicted_nf_hectares = nf_count * hectares_per_pixel

    # Get an array of the forested pixels, scaled to correct range
    f_arr = np.ma.where(arr >= 0, arr / scalar, np.ma.masked).compressed()

    # Return the forested array and the nonforest pixel area
    return f_arr, predicted_nf_hectares, hectares_per_pixel


class RegionalAccuracyDiagnostic(diagnostic.Diagnostic):
    _required = [
        "area_estimate_file",
        "stand_attribute_file",
        "stand_metadata_file",
    ]

    def __init__(self, parameter_parser):
        self.parameter_parser = parameter_parser
        self.id_field = parameter_parser.plot_id_field

        # We want to check on the existence of a few files
        # before running, so create some instance variables
        self.area_estimate_file = parameter_parser.area_estimate_file
        self.stand_attribute_file = parameter_parser.stand_attribute_file
        self.stand_metadata_file = parameter_parser.stand_metadata_file

        self.check_missing_files()

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        return cls(parameter_parser)

    def get_observed_estimates(self):
        # Read the area estimate file into a recarray
        obs_df = pd.read_csv(self.area_estimate_file)

        # Get the nonforest hectares (coded as -10001)
        nf_row = obs_df[obs_df[self.id_field] == -10001]
        nf_hectares = nf_row.at[nf_row.index[0], "HECTARES"]

        # Get the nonsampled hectares (coded as -10002)
        ns_row = obs_df[obs_df[self.id_field] == -10002]
        ns_hectares = ns_row.at[ns_row.index[0], "HECTARES"]

        # Remove these rows from the recarray
        obs_df = obs_df[obs_df[self.id_field] > 0]

        # Return this information
        return obs_df, nf_hectares, ns_hectares

    @staticmethod
    def insert_class(hist, name, count):
        hist.bin_counts = np.insert(hist.bin_counts, [0], count)
        hist.bin_names.insert(0, name)
        return hist

    @staticmethod
    def _bin_data(attr, obs, prd, bin_type="EQUAL_INTERVAL", bin_count=7):
        if attr.is_continuous():
            # The obs and prd parameters are either arrays or WeightedArrays.
            # For the purpose of getting the classifier, first convert to
            # flattened arrays
            obs_arr = obs.flatten() if isinstance(obs, WeightedArray) else obs
            prd_arr = prd.flatten() if isinstance(prd, WeightedArray) else prd

            # Get the classifier.  For equal interval classification, use
            # both observed and predicted to determine the full range of values.
            # For quantile classification, use only the observed values to
            # set the bins
            if bin_type == "EQUAL_INTERVAL":
                clf = ic.EqualIntervals(bin_count=bin_count)
                bins = clf(np.hstack((obs_arr, prd_arr)))
            elif bin_type == "NATURAL_BREAKS":
                # This is a hack to make this method tractable.  First,
                # stack the observed and predicted arrays, sort it, and
                # then subset to subset_size elements, making sure to include
                # the min and max of the array.
                subset_size = 40000
                a = np.hstack((obs_arr, prd_arr))
                a.sort()
                step = max(1, int(a.size / subset_size))
                arr = np.hstack((a.min(), a[step // 2 :: step], a.max()))
                clf = ic.NaturalBreaksIntervals(bin_count=bin_count)
                bins = clf(arr)
            elif bin_type == "QUANTILE":
                clf = ic.QuantileIntervals(bin_count=bin_count)
                bins = clf(obs_arr)
            else:
                msg = f"Bin type {bin_type} not supported"
                raise ValueError(msg)

            # For getting the histograms, use the actual obs and prd objects
            histograms = histogram.bin_continuous(*(obs, prd), bins=bins)
        else:
            if attr.codes:
                code_dict = {
                    int(code.code_value): code.label for code in attr.codes
                }
            else:
                code_dict = None
            histograms = histogram.bin_categorical(
                *(obs, prd), code_dict=code_dict
            )

        histograms[0].name = "OBSERVED"
        histograms[1].name = "PREDICTED"
        return histograms

    def create_paired_area_file(self):
        # Read in the observed data from the area estimate file
        obs_area, obs_nf_ha, obs_ns_ha = self.get_observed_estimates()

        # Get the weights for the observed data
        obs_weights = obs_area.HECTARES

        # Open the output file and print out the header line
        statistics_file = self.parameter_parser.regional_accuracy_file
        stats_fh = open(statistics_file, "w")
        header_fields = ["VARIABLE", "DATASET", "BIN_NAME", "AREA"]
        stats_fh.write(",".join(header_fields) + "\n")

        # Open the classification bin file and print out the header line
        bin_file = self.parameter_parser.regional_bin_file
        bin_fh = open(bin_file, "w")
        bin_fh.write(f"VARIABLE,CLASS,LOW,HIGH\n")

        # Get the metadata parser and get the project area attributes
        mp = XMLStandMetadataParser(self.stand_metadata_file)
        attrs = mp.filter(
            Flags.CONTINUOUS
            | Flags.ACCURACY
            | Flags.PROJECT
            | Flags.NOT_SPECIES
        )
        attrs.extend(
            mp.filter(
                Flags.CATEGORICAL
                | Flags.ACCURACY
                | Flags.PROJECT
                | Flags.NOT_SPECIES
            )
        )

        # Iterate over all fields and print out the area histogram statistics
        for attr in attrs:
            print(attr.field_name)

            # Observed values and weights for this field
            obs_vals = obs_area[attr.field_name]
            obs_wa = WeightedArray(obs_vals, obs_weights)

            # Ensure the predicted raster has been created
            fn = f"./attribute_rasters/{attr.field_name.lower()}.tif"
            if not os.path.exists(fn):
                print(f"Building {attr.field_name} raster ...")
                build_attribute_raster.main(
                    self.parameter_parser, attr.field_name
                )

            # Get predicted areas from predicted rasters that should be
            # pre-generated
            prd_ns_ha = 0.0
            prd_f_arr, prd_nf_ha, ha_per_px = get_predicted_raster(attr)
            needs_conversion = (False, True)

            # Bin the data based on field type
            bins = self._bin_data(
                attr,
                obs_wa,
                prd_f_arr,
                bin_type=self.parameter_parser.error_matrix_bin_method,
                bin_count=self.parameter_parser.error_matrix_bin_count,
            )

            # Write out the classification bins - these will be used later
            # in constructing the error matrix for Olofsson error-based
            # area adjustment
            obs_bins = bins[0]
            for i in range(len(obs_bins.bin_endpoints) - 1):
                out_list = [
                    attr.field_name,
                    "{:d}".format(i + 1),
                    "{:.4f}".format(obs_bins.bin_endpoints[i]),
                    "{:.4f}".format(obs_bins.bin_endpoints[i + 1]),
                ]
                bin_fh.write(",".join(out_list) + "\n")

            # If bins need to be converted to hectares, do that here
            for d, convert_flag in zip(bins, needs_conversion):
                if convert_flag:
                    d.bin_counts = np.multiply(d.bin_counts, ha_per_px)

            # Handle special cases of nonsampled and nonforest area
            self.insert_class(bins[0], "Unsampled", obs_ns_ha)
            self.insert_class(bins[0], "Nonforest", obs_nf_ha)
            self.insert_class(bins[1], "Unsampled", prd_ns_ha)
            self.insert_class(bins[1], "Nonforest", prd_nf_ha)

            for b in bins:
                for i in range(len(b.bin_counts)):
                    out_data = "{:s},{:s},{:s},{:.3f}\n".format(
                        attr.field_name, b.name, b.bin_names[i], b.bin_counts[i]
                    )
                    stats_fh.write(out_data)

    def exclude_forest_minority(self, current_ids):
        ae_df = pd.read_csv(
            self.area_estimate_file, usecols=[self.id_field, "PNT_COUNT"]
        )
        forest_majority = ae_df[ae_df.PNT_COUNT >= 2.0]
        return list(set(current_ids) & set(forest_majority[self.id_field]))

    def create_predicted_plot_file(self):
        parser = self.parameter_parser

        # Create a dictionary of plot ID to image year for these plots
        ids = get_id_list(self.area_estimate_file, self.id_field)

        # Exclude any plots without at least half of points in forested
        # condition (< 2.0)
        ids = self.exclude_forest_minority(ids)

        # Create a PixelNNFinder object and calculate neighbors and distances
        finder = PixelNNFinder(parser)
        neighbor_data = finder.calculate_neighbors_at_ids(ids)

        # Get predicted attributes
        output = IndependentOutput(parser, neighbor_data)
        output.write_attribute_predictions(parser.regional_predicted_plot_file)

    def create_error_matrix_file(self):
        parser = self.parameter_parser
        emd = ErrorMatrixDiagnostic(
            parser.area_estimate_file,
            parser.regional_predicted_plot_file,
            parser.stand_metadata_file,
            parser.plot_id_field,
            parser.regional_error_matrix_file,
            input_bin_file=parser.regional_bin_file,
        )
        emd.run_diagnostic()

    def create_olofsson_file(self):
        parser = self.parameter_parser
        olofsson = OlofssonDiagnostic(
            parser.regional_error_matrix_file,
            parser.regional_bin_file,
            parser.regional_accuracy_file,
            parser.stand_metadata_file,
            parser.regional_olofsson_file,
        )
        olofsson.run_diagnostic()

    def run_diagnostic(self):
        self.create_paired_area_file()
        self.create_predicted_plot_file()
        self.create_error_matrix_file()
        self.create_olofsson_file()
