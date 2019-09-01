import os

import pandas as pd
from tqdm import tqdm

from .feature import FeatureFactory


class DataProcessor:
    def __init__(self, config):
        self.pickled_feature_dir = config.pickled_feature_dir
        self.feature_names = config.feature_names
        self.target_name = config.target_name

    def run(self, dataset):
        X_train = self._make_X(dataset.train, "train")
        X_test = self._make_X(dataset.test, "test")
        y_train = dataset.train[self.target_name]
        return X_train, y_train, X_test

    def _make_X(self, df, kind):
        ff = FeatureFactory()
        features = [ff(name) for name in self.feature_names]
        X = pd.DataFrame(index=df.index)
        for feature in tqdm(features):
            name = feature.__class__.__name__
            path = os.path.join(self.pickled_feature_dir, f"{kind}", f"{name}.pkl")
            if os.path.isfile(path):
                values = self.load_feature(path)
            else:
                values = pd.DataFrame(feature.run(df))
                self._save_feature(values, path)
                print(f"save {name} feature for {kind} to pickle")
            X = X.join(values)
        return X

    @staticmethod
    def _save_feature(values, path):
        values.to_pickle(path)

    @classmethod
    def load_feature(cls, path):
        return pd.read_pickle(path)
