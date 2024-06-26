from __future__ import annotations

from itertools import cycle

import numpy as np
import pandas as pd

from ..parser.xml_stand_metadata_parser import Flags, XMLStandMetadataParser
from . import diagnostic


def standardize_by_axis(arr, axis=0):
    axis_sums = arr.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_ij = arr / axis_sums
        n_ij[~np.isfinite(n_ij)] = 0.0
    return n_ij


class ErrorAdjustment:
    def __init__(self, error_matrix, mapped_areas):
        self.error_matrix = np.array(error_matrix, dtype=np.float64)
        self.mapped_areas = np.array(mapped_areas)
        if self.mapped_areas.shape[0] != 1:
            self.mapped_areas = self.mapped_areas[np.newaxis, :].T
        self.w = self.mapped_areas / self.mapped_areas.sum()
        self.n_ij = standardize_by_axis(self.error_matrix, axis=1)
        self.p_ij = np.multiply(self.w, self.n_ij)

    def adjusted_areas(self):
        return self.p_ij.sum(axis=0) * self.mapped_areas.sum()

    def confidence_intervals(self):
        row_sums = self.error_matrix.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 2, row_sums.max())
        se_ij = ((self.n_ij * (1.0 - self.n_ij)) / (row_sums - 1)) * (self.w * self.w)
        return 2.0 * np.sqrt(se_ij.sum(axis=0)) * self.mapped_areas.sum()


class OlofssonDiagnostic(diagnostic.Diagnostic):
    _required: list[str] = [
        "error_matrix_file",
        "bin_file",
        "regional_accuracy_file",
        "stand_metadata_file",
    ]

    def __init__(
        self,
        error_matrix_file,
        bin_file,
        regional_accuracy_file,
        stand_metadata_file,
        olofsson_file,
    ):
        self.error_matrix_file = error_matrix_file
        self.bin_file = bin_file
        self.regional_accuracy_file = regional_accuracy_file
        self.stand_metadata_file = stand_metadata_file
        self.olofsson_file = olofsson_file

        self.check_missing_files()

        self.error_matrix_df = pd.read_csv(self.error_matrix_file)
        self.bin_df = pd.read_csv(self.bin_file)
        self.regional_df = pd.read_csv(self.regional_accuracy_file)

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        p = parameter_parser
        return cls(
            p.error_matrix_file,
            p.bin_file,
            p.regional_accuracy_file,
            p.stand_metadata_file,
            p.olofsson_file,
        )

    def attr_missing(self, attr):
        if attr.field_name not in self.error_matrix_df.VARIABLE.values:
            return True
        return attr.field_name not in self.regional_df.VARIABLE.values

    def run_attr(self, attr):
        fn = attr.field_name
        em_df = self.error_matrix_df
        area_df = self.regional_df
        em_data = em_df[fn == em_df.VARIABLE]
        err_matrix = pd.pivot_table(
            em_data,
            index="PREDICTED_CLASS",
            columns="OBSERVED_CLASS",
            values="COUNT",
        )
        conds = (fn == area_df.VARIABLE) & (area_df.DATASET == "PREDICTED")
        bin_names = np.array(area_df[conds].BIN_NAME.iloc[2:])
        mapped = np.array(area_df[conds].AREA.iloc[2:])
        olofsson = ErrorAdjustment(err_matrix, mapped)
        return [fn, bin_names, mapped, olofsson]

    @staticmethod
    def write_records(fh, attr_name, bin_names, mapped, adjusted, se_adjusted):
        for record in zip(cycle([attr_name]), bin_names, mapped, adjusted, se_adjusted):
            fh.write("{:s},{:s},{:.4f},{:.4f},{:.4f}\n".format(*record))

    def run_diagnostic(self):
        # Read in the stand attribute metadata and get continuous
        # and categorical attributes
        mp = XMLStandMetadataParser(self.stand_metadata_file)
        attrs = mp.filter(
            Flags.CONTINUOUS | Flags.ACCURACY | Flags.PROJECT | Flags.NOT_SPECIES
        )
        attrs.extend(mp.filter(Flags.CATEGORICAL | Flags.ACCURACY | Flags.PROJECT))

        # For each attribute, calculate the statistics
        statistics_data = []
        for attr in attrs:
            if self.attr_missing(attr):
                continue
            statistics_data.append(self.run_attr(attr))

        with open(self.olofsson_file, "w") as olofsson_fh:
            olofsson_fh.write("VARIABLE,CLASS,MAPPED,ADJUSTED,CI_ADJUSTED\n")
            for data in statistics_data:
                attr_name, bin_names, mapped, olofsson = data
                self.write_records(
                    olofsson_fh,
                    attr_name,
                    bin_names,
                    mapped,
                    olofsson.adjusted_areas(),
                    olofsson.confidence_intervals(),
                )
