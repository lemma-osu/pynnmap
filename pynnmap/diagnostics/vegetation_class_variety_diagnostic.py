import pandas as pd

from pynnmap.diagnostics import diagnostic
from pynnmap.misc.utilities import df_to_csv

COARSE_VC_REMAP = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 1,
    6: 2,
    7: 3,
    8: 1,
    9: 2,
    10: 3,
    11: 3,
}


def calculate_vc_variety(vc_records):
    # Remap the vc_records to coarser categories
    # - Open/young (VCs 1, 2, 3, 5, 8)
    # - Closed/medium (VCs 4, 6, 9)
    # - Closed/large (VCs 7, 10, 11)
    coarse_records = [COARSE_VC_REMAP[x] for x in vc_records]

    # Identify if this plot should be considered a variety plot:
    # has at least 2 '1's and 2 '3's in it
    young = [x for x in coarse_records if x == 1]
    old = [x for x in coarse_records if x == 3]
    return True if len(young) >= 2 and len(old) >= 2 else False


class VegetationClassVarietyDiagnostic(diagnostic.Diagnostic):
    _required = [
        "stand_attr_file",
        "dependent_zonal_pixel_file",
        "independent_zonal_pixel_file",
    ]

    def __init__(self, parameters):
        p = parameters
        self.stand_attr_file = p.stand_attribute_file
        self.id_field = p.plot_id_field
        self.output_file = p.vegclass_variety_file

        # Create a list of zonal_pixel files - both independent and dependent
        self.dependent_zonal_pixel_file = p.dependent_zonal_pixel_file
        self.independent_zonal_pixel_file = p.independent_zonal_pixel_file
        self.zonal_pixel_files = [
            ("dependent", self.dependent_zonal_pixel_file),
            ("independent", self.independent_zonal_pixel_file),
        ]

        self.check_missing_files()

    def _vc_variety(self, rec, zonal_df):
        cond = zonal_df[self.id_field] == rec[self.id_field]
        records = zonal_df[cond].VEGCLASS
        return calculate_vc_variety(records)

    def run_diagnostic(self):
        # Open the stand attribute file and subset to just positive IDs
        columns = [self.id_field, "VEGCLASS"]
        attr_df = pd.read_csv(self.stand_attr_file, usecols=columns)
        attr_df = attr_df[attr_df[self.id_field] > 0]

        # Run this for both independent and dependent predictions
        dfs = []
        for (prd_type, zp_file) in self.zonal_pixel_files:
            # Create a copy of the attr_df for this prd_type and insert a
            # column for this
            df = attr_df.copy()
            df.insert(1, "PREDICTION_TYPE", prd_type.upper())

            # Open the zonal pixel file and join vegclass to it
            zonal_df = pd.read_csv(zp_file)
            zonal_df = zonal_df.merge(
                df,
                left_on="NEIGHBOR_ID",
                right_on=self.id_field,
                suffixes=["", "_DUP"],
            )

            # Calculate the vc_variety
            df["OUTLIER"] = df.apply(self._vc_variety, axis=1, args=(zonal_df,))

            # Save out the records that are True
            df = df[df.OUTLIER]
            dfs.append(df[[self.id_field, "PREDICTION_TYPE"]])

        # Merge together the dfs and export
        out_df = pd.concat(dfs)
        df_to_csv(out_df, self.output_file)
