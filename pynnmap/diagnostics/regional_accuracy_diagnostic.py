import numpy as np
import pandas as pd
import rasterio
from matplotlib import mlab
from osgeo import gdal, gdalconst

from pynnmap.core import (
    get_id_year_crosswalk,
    get_independence_filter,
    get_weights,
    get_id_list,
)
from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.core.nn_finder import NNFinder
from pynnmap.core.attribute_predictor import AttributePredictor
from pynnmap.diagnostics import diagnostic
from pynnmap.diagnostics.error_matrix_diagnostic import ErrorMatrixDiagnostic
from pynnmap.diagnostics.olofsson_diagnostic import OlofssonDiagnostic
from pynnmap.misc import histogram
from pynnmap.misc import interval_classification as ic
from pynnmap.misc import utilities
from pynnmap.misc.utilities import df_to_csv
from pynnmap.misc.weighted_array import WeightedArray
from pynnmap.parser.xml_parameter_parser import XMLParameterParser
from pynnmap.parser.xml_stand_metadata_parser import XMLStandMetadataParser
from pynnmap.parser.xml_stand_metadata_parser import Flags


def get_predicted_raster(attr):
    # TODO: This is *very* fragile
    fn = "./attribute_rasters/{}.tif".format(attr.field_name.lower())
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

    def get_predicted_estimates(self):
        # Read in the predicted raster
        ds = gdal.Open(self.predicted_raster, gdalconst.GA_ReadOnly)
        rb = ds.GetRasterBand(1)

        # Get the cell area for converting from pixel counts to hectares
        gt = ds.GetGeoTransform()
        cell_area = gt[1] * gt[1]
        hectares_per_pixel = cell_area / 10000.0

        # Calculate statistics directly on the raster
        z_min, z_max = int(rb.GetMinimum()), int(rb.GetMaximum())
        z_range = z_max - z_min + 1
        hist = rb.GetHistogram(z_min - 0.5, z_max + 0.5, z_range, False, False)
        bins = list(range(z_min, z_max + 1))
        rat = [(x, y) for x, y in zip(bins, hist) if y != 0]

        # Get the IDs and counts (converted to hectares)
        id_recs = []
        nf_hectares = 0
        for (id_val, count) in rat:
            hectares = count * hectares_per_pixel
            if id_val <= 0:
                nf_hectares += hectares
            else:
                id_recs.append((id_val, hectares))

        # Release the dataset
        del ds

        # Convert this to a recarray
        names = (self.id_field, "HECTARES")
        ids = np.rec.fromrecords(id_recs, names=names)

        # Read in the attribute file
        sad = pd.read_csv(self.stand_attribute_file)

        # Ensure that all IDs in the id_count_dict are in the attribute data
        ids_1 = getattr(ids, self.id_field)
        ids_2 = getattr(sad, self.id_field)
        if not np.all(np.in1d(ids_1, ids_2)):
            err_msg = "Not all values in the raster are present in the "
            err_msg += "attribute data"
            raise ValueError(err_msg)

        # Join the two recarrays together
        predicted_data = mlab.rec_join(self.id_field, ids, sad)
        return predicted_data, nf_hectares

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
            elif bin_type == "QUANTILE":
                clf = ic.QuantileIntervals(bin_count=bin_count)
                bins = clf(obs_arr)
            else:
                msg = "Bin type {} not supported".format(bin_type)
                raise ValueError(msg)

            # For getting the histograms, use the actual obs and prd objects
            histograms = histogram.bin_continuous(*(obs, prd), bins=bins)
        else:
            if attr.codes:
                class_names = {}
                for code in attr.codes:
                    class_names[int(code.code_value)] = code.label
            else:
                class_names = None
            histograms = histogram.bin_categorical(
                *(obs, prd), class_names=class_names
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
        bin_fh.write("{},{},{},{}\n".format("VARIABLE", "CLASS", "LOW", "HIGH"))

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

            # Getting predicted areas differ based on how many neighbors are
            # used.  If using k=1, we can use a lookup table to get pixel
            # areas for imputed plots and crosswalk to attributes; otherwise
            # we need to pre-calculate the attribute raster and obtain values
            # from it
            prd_ns_ha = 0.0
            prd_nf_ha = 0.0
            ha_per_px = 0.0
            if self.parameter_parser.k == 1:
                # prd_area, prd_nf_ha = self.get_predicted_estimates()
                # prd_weights = prd_area.HECTARES
                # prd_wa = histogram.WeightedArray(prd_vals, prd_weights)
                # datasets = (obs_wa, prd_wa)
                # prd_f_arr = None
                # datasets = (obs_wa, prd)
                # needs_conversion = (False,)
                pass
            else:
                prd_f_arr, prd_nf_ha, ha_per_px = get_predicted_raster(attr)
                # datasets = (obs_wa, prd_f_arr)
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
                for i in range(0, len(b.bin_counts)):
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

        # Get the crosswalk to assessment year
        id_x_year = get_id_year_crosswalk(parser)
        id_x_year = dict((k, v) for k, v in id_x_year.items() if k in ids)

        # Create a NNFinder object and calculate neighbors and distances
        finder = NNFinder(parser)
        neighbor_data = finder.calculate_neighbors_at_ids(id_x_year)

        # Create an independence filter based on the relationship of the
        # id_field and the no_self_assign_field
        fltr = get_independence_filter(parser)

        # Get the stand attributes and filter to continuous accuracy fields
        mp = XMLStandMetadataParser(self.stand_metadata_file)
        model_attr_fn = parser.stand_attribute_file
        model_attr_data = StandAttributes(
            model_attr_fn, mp, id_field=self.id_field
        )
        plot_attr_predictor = AttributePredictor(model_attr_data, fltr)

        # Calculate the predictions for each plot
        predictions = plot_attr_predictor.calculate_predictions(
            neighbor_data, k=parser.k, weights=get_weights(parser)
        )

        # Write out predicted attribute file
        prd_df = plot_attr_predictor.get_predicted_attributes_df(
            predictions, self.id_field
        )
        df_to_csv(prd_df, parser.regional_predicted_plot_file, index=True)

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
