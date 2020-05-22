from pathlib import Path

import pandas as pd

SCALAR_FN = Path(__file__).parent.joinpath("scalars.csv")


def get_scalar(attr):
    scalars_df = pd.read_csv(SCALAR_FN, index_col="ATTR")
    scalars = scalars_df.to_dict()["SCALAR"]
    return scalars[attr.upper()]


def get_k(attr):
    scalars_df = pd.read_csv(SCALAR_FN, index_col="ATTR")
    scalars = scalars_df.to_dict()["K"]
    return int(scalars[attr.upper()])
