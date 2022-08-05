import pandas as pd


class StandAttributes(object):
    def __init__(self, fn, metadata, id_field='FCID', **kwargs):
        self._df = pd.read_csv(
            fn, low_memory=False, index_col=id_field, **kwargs)
        self._metadata = metadata
        attr_names = set(list(self._df.columns) + [id_field])
        metadata_names = set(self._metadata.attr_names())
        if missing := attr_names - metadata_names:
            msg = f'Metadata fields {tuple(missing)} are missing'
            raise ValueError(msg)

    def get_attr_df(self, flags=None):
        attr_cols = self._metadata.filter(flags=flags)
        if self.df.index.name in attr_cols:
            df = self.df.copy(deep=True)
            df[df.index.name] = df.index
        else:
            df = self.df
        subset = set(df.columns) & set(attr_cols)
        return df[sorted(subset, key=list(df.columns).index)]

    @property
    def df(self):
        return self._df
