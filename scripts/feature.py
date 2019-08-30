import inspect

import pandas as pd

from .config import Config


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        if feature_names in globals():
            return globals()[feature_names](**kwargs)
        else:
            raise ValueError("No feature defined named with {}".format(feature_names))

    def feature_list(self):
        lst = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                Config,
                FeatureFactory,
                Feature,
                BasicFeature,
            ]:
                lst.append(obj.__name__)
        return lst


class Feature:
    def __init___(self, **kwargs):
        self.name = str(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def run(self, df):
        values = self.extract(df)
        values = values.fillna(0)
        return values

    def extract(self, df):
        raise NotImplementedError

    @staticmethod
    def _get_converted_multi_columns(values, head_name=None):
        col_names = [col[0] + "_" + col[1] for col in values.columns.values]
        if head_name:
            col_names = [head_name + "_" + col for col in col_names]
        return col_names


class BasicFeature(Feature):

    column = None
    prefix = None

    def extract(self, df):
        values = df[self.column]
        if values.dtype == "O":
            values = pd.get_dummies(values, prefix=self.prefix)
        return values
