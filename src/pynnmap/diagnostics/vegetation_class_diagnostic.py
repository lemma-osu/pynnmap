from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..misc import classification_accuracy as ca
from ..misc.utilities import df_to_csv
from . import diagnostic

BASEDIR = os.path.abspath(os.path.dirname(__file__))
VEGCLASS_XML = os.path.join(BASEDIR, "vegclass.xml")


class VegetationClassDiagnostic(diagnostic.Diagnostic):
    _required: list[str] = ["observed_file", "predicted_file"]

    qmd_breaks = np.array([0.0, 2.5, 25.0, 37.5, 50.0, 75.0])
    qmd_classes = np.array([1, 2, 3, 4, 5, 6])

    cancov_breaks = np.array([0.0, 10.0, 40.0, 70.0])
    cancov_classes = np.array([1, 2, 3, 4])

    def __init__(
        self,
        observed_file,
        predicted_file,
        id_field,
        vegclass_file,
        vegclass_kappa_file,
        vegclass_errmatrix_file,
    ):
        self.observed_file = observed_file
        self.predicted_file = predicted_file
        self.id_field = id_field
        self.vegclass_file = vegclass_file
        self.vegclass_kappa_file = vegclass_kappa_file
        self.vegclass_errmatrix_file = vegclass_errmatrix_file

        self.check_missing_files()

    @classmethod
    def from_parameter_parser(cls, parameter_parser):
        p = parameter_parser
        return cls(
            p.stand_attribute_file,
            p.independent_predicted_file,
            p.plot_id_field,
            p.vegclass_file,
            p.vegclass_kappa_file,
            p.vegclass_errmatrix_file,
        )

    def get_vegclass(self, rec, field_dict):
        """
        Return the VEGCLASS for this plot given the input data
        """

        qmd = getattr(rec, field_dict["qmd"])
        cancov = getattr(rec, field_dict["cancov"])
        bah = getattr(rec, field_dict["bah"])

        # Size class based on qmd
        index = np.digitize([qmd], self.qmd_breaks)
        sc = self.qmd_classes[index - 1][0]

        # Cover class based on cancov
        index = np.digitize([cancov], self.cancov_breaks)
        cc = self.cancov_classes[index - 1][0]

        # Step through logic to determine VEGCLASS
        if cc == 1:
            vc = 1
        elif cc == 2:
            vc = 2
        elif bah >= 0.65:
            vc = 3 if sc <= 2 else 4
        elif 0.2 <= bah < 0.65:
            if sc <= 2:
                vc = 5
            elif 3 <= sc <= 4:
                vc = 6
            else:
                vc = 7
        elif sc <= 2:
            vc = 8
        elif 3 <= sc <= 4:
            vc = 9
        elif sc == 5:
            vc = 10
        else:
            vc = 11
        return vc

    def vegclass_aa(
        self,
        obs,
        prd,
        id_field="FCID",
        qmd_field="QMD_DOM",
        bah_field="BAH_PROP",
        cancov_field="CANCOV",
    ):
        """
        Given observed and predicted recarrays from GNN output, this
        function creates a data file of observed and predicted vegetation
        class (VEGCLASS)
        """

        # Create a dict of needed field names
        field_dict = {
            "id": id_field,
            "qmd": qmd_field,
            "bah": bah_field,
            "cancov": cancov_field,
        }

        # Ensure that the fields identified exist in the recarrays
        for field in field_dict.values():
            if field not in obs.columns:
                raise ValueError("Cannot find " + field + " in observed file")
            if field not in prd.columns:
                raise ValueError("Cannot find " + field + " in predicted file")

        # Get the vegetation class for every observed and predicted record
        # and merge together into a data frame
        obs["VC"] = obs.apply(self.get_vegclass, axis=1, args=(field_dict,))
        prd["VC"] = prd.apply(self.get_vegclass, axis=1, args=(field_dict,))
        obs_vc = obs[[id_field, "VC"]]
        prd_vc = prd[[id_field, "VC"]]
        return obs_vc.merge(prd_vc, on=id_field)

    def run_diagnostic(self):
        # Read the observed and predicted files into numpy recarrays
        obs = pd.read_csv(self.observed_file, low_memory=False)
        prd = pd.read_csv(self.predicted_file, low_memory=False)

        # Subset the observed data just to the IDs that are in the
        # predicted file
        obs_keep = np.in1d(getattr(obs, self.id_field), getattr(prd, self.id_field))
        obs = obs[obs_keep]

        # Calculate VEGCLASS for both the observed and predicted data
        vc_df = self.vegclass_aa(obs, prd, id_field=self.id_field)
        vc_df.columns = [self.id_field, "OBSERVED", "PREDICTED"]
        df_to_csv(vc_df, self.vegclass_file)

        # Create the vegetation class kappa and error matrix files
        ca.classification_accuracy(
            self.vegclass_file,
            VEGCLASS_XML,
            kappa_fn=self.vegclass_kappa_file,
            err_matrix_fn=self.vegclass_errmatrix_file,
        )
