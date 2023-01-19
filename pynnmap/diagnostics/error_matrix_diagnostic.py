import itertools
import numpy as np
import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import interval_classification as ic
from pynnmap.misc import utilities
from pynnmap.parser.xml_stand_metadata_parser import XMLStandMetadataParser
from pynnmap.parser.xml_stand_metadata_parser import Flags


def create_existing_bins(bin_file):
    def get_custom_classifier(group_df):
        endpoints = np.hstack((group_df.LOW.values, group_df.HIGH.values[-1]))
        return ic.CustomIntervals(endpoints)

    # Read in the bins from an existing file
    clf_dict = {}
    df = pd.read_csv(bin_file)
    grouped_df = df.groupby("VARIABLE")
    for name, group in grouped_df:
        clf_dict[name] = get_custom_classifier(group)
    return clf_dict


def get_error_matrix(obs_vals, prd_vals, intervals, return_bins=True):
    digitizer = ic.DataDigitizer(intervals)
    digitizer.set_bins(np.hstack((obs_vals, prd_vals)))
    obs, prd = map(digitizer.bin_data, (obs_vals, prd_vals))
    err_mat = pd.crosstab(index=obs, columns=prd, dropna=False)
    return (err_mat, digitizer.bins) if return_bins else err_mat


class ErrorMatrixDiagnostic(diagnostic.Diagnostic):
    _required = ["observed_file", "predicted_file", "stand_metadata_file"]

    _classifier = {
        "EQUAL_INTERVAL": ic.EqualIntervals,
        "QUANTILE": ic.QuantileIntervals,
        "NATURAL_BREAKS": ic.NaturalBreaksIntervals,
    }

    def __init__(
        self,
        observed_file,
        predicted_file,
        stand_metadata_file,
        id_field,
        error_matrix_file,
        classifier=None,
        input_bin_file=None,
        output_bin_file=None,
    ):
        self.observed_file = observed_file
        self.predicted_file = predicted_file
        self.id_field = id_field
        self.stand_metadata_file = stand_metadata_file
        self.error_matrix_file = error_matrix_file

        if classifier is not None:
            self.clf = classifier

        if input_bin_file is not None:
            self.clf = None
            self.clf_dict = create_existing_bins(input_bin_file)

        self.output_bin_file = None
        if output_bin_file is not None:
            self.output_bin_file = output_bin_file

        self.check_missing_files()
        self.obs_df, self.prd_df = utilities.build_obs_prd_dataframes(
            self.observed_file, self.predicted_file, self.id_field
        )

    @classmethod
    def from_parameter_parser(cls, parameter_parser, **kwargs):
        # Get the classifier before creating the instance - if a bin_file
        # is passed in, use defined bins rather than dynamic bins
        if "bin_file" in kwargs and kwargs["bin_file"] is not None:
            clf = None
            input_bin_file = kwargs["bin_file"]
            output_bin_file = None
        else:
            bin_method = parameter_parser.error_matrix_bin_method
            bin_count = parameter_parser.error_matrix_bin_count
            clf = cls._classifier[bin_method](bin_count)
            input_bin_file = None
            output_bin_file = parameter_parser.error_matrix_bin_file

        return cls(
            parameter_parser.stand_attribute_file,
            parameter_parser.independent_predicted_file,
            parameter_parser.stand_metadata_file,
            parameter_parser.plot_id_field,
            parameter_parser.error_matrix_accuracy_file,
            classifier=clf,
            input_bin_file=input_bin_file,
            output_bin_file=output_bin_file,
        )

    def attr_missing(self, attr):
        if attr.field_name not in self.obs_df.columns:
            return True
        return attr.field_name not in self.prd_df.columns

    def run_attr(self, attr, clf, return_bins=True):
        obs_vals = getattr(self.obs_df, attr.field_name)
        prd_vals = getattr(self.prd_df, attr.field_name)
        return get_error_matrix(
            obs_vals, prd_vals, clf, return_bins=return_bins
        )

    def run_diagnostic(self):
        # Open the error matrix file and print out the header line
        err_matrix_fh = open(self.error_matrix_file, "w")
        err_matrix_fh.write(f"VARIABLE,OBSERVED_CLASS,PREDICTED_CLASS,COUNT\n")

        # Open the bin file and print out the header line
        if self.output_bin_file:
            bin_fh = open(self.output_bin_file, "w")
            bin_fh.write(f"VARIABLE,CLASS,LOW,HIGH\n")

        # Read in the stand attribute metadata and get continuous
        # and categorical attributes
        mp = XMLStandMetadataParser(self.stand_metadata_file)
        attrs = mp.filter(Flags.CONTINUOUS | Flags.ACCURACY)
        attrs.extend(mp.filter(Flags.CATEGORICAL | Flags.ACCURACY))

        # For each attribute, calculate the statistics
        for attr in attrs:
            if self.attr_missing(attr):
                continue

            if attr.field_type == "CATEGORICAL":
                bins = sorted([int(x.code_value) for x in attr.codes])
                bins.append(bins[-1] + 1)
                clf = ic.CustomIntervals(bins)
                err_mat, bins = self.run_attr(attr, clf, return_bins=True)
            elif self.clf is not None:
                err_mat, bins = self.run_attr(attr, self.clf, return_bins=True)
            else:
                if attr.field_name not in self.clf_dict:
                    continue
                clf = self.clf_dict[attr.field_name]
                err_mat = self.run_attr(attr, clf, return_bins=False)

            rows, cols = err_mat.shape
            labels = list(err_mat.index)
            for j, i in itertools.product(range(cols), range(rows)):
                out_list = [
                    attr.field_name,
                    f"{labels[i]}",
                    f"{labels[j]}",
                    f"{err_mat.iat[i, j]}",
                ]
                err_matrix_fh.write(",".join(out_list) + "\n")

            if self.output_bin_file:
                for i in range(len(labels)):
                    try:
                        start, end = bins[i], bins[i + 1]
                    except IndexError:
                        start, end = bins[i], bins[i] + 1
                    out_list = [
                        attr.field_name,
                        f"{labels[i]}",
                        "{:.4f}".format(start),
                        "{:.4f}".format(end),
                    ]
                    bin_fh.write(",".join(out_list) + "\n")
