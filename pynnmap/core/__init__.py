import numpy as np
import pandas as pd
from pynnmap.core.independence_filter import IndependenceFilter


def get_id_year_crosswalk(parser):
    id_field = parser.plot_id_field
    xwalk_df = pd.read_csv(parser.plot_year_crosswalk_file, low_memory=False)
    if parser.model_type in parser.imagery_model_types:
        s = pd.Series(xwalk_df.IMAGE_YEAR.values, index=xwalk_df[id_field])
    else:
        s = pd.Series(parser.model_year, index=xwalk_df[id_field])
    return dict(s.to_dict())


def get_independence_filter(parser, no_self_assign_field='LOC_ID'):
    id_field = parser.plot_id_field
    fn = parser.plot_independence_crosswalk_file
    fields = [id_field, no_self_assign_field]
    df = pd.read_csv(fn, usecols=fields, index_col=id_field)
    return IndependenceFilter.from_common_lookup(
        df.index, df[no_self_assign_field]
    )


def get_weights(parser):
    w = parser.weights
    if w is not None:
        if len(w) != parser.k:
            raise ValueError('Length of weights does not equal k')
        w = np.array(w).reshape(1, len(w)).T
    return w


def get_id_list(attr_fn, id_field):
    df = pd.read_csv(attr_fn, usecols=(id_field,))
    return df[df[id_field] >= 0][id_field].tolist()
