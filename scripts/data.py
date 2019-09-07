import os

import pandas as pd
from collections import namedtuple

from .db import LocalFile
from .preprocess import Preprocessor


class DatasetCreator:
    def __init__(self, config):
        self.config = config

    def run(self):
        if os.path.isfile(self.config.pickled_train_path):
            # load pickled files
            train = pd.read_pickle(self.config.pickled_train_path)
            print("Succeeded in loading pickled train data")
            test = pd.read_pickle(self.config.pickled_test_path)
            print("Succeeded in loading pickled test data")

        else:
            # load files
            db = LocalFile(self.config)
            train_t = db.get_train_transaction()
            print("Succeeded in loading train_transaction.csv")
            test_t = db.get_test_transaction()
            print("Succeeded in loading train_transaction.csv")
            train_i = db.get_train_identity()
            print("Succeeded in loading train_identity.csv")
            test_i = db.get_test_identity()
            print("Succeeded in loading test_identity.csv")

            # preprocess data
            preprocessor = Preprocessor(self.config)
            train, test = preprocessor.run(train_t, test_t, train_i, test_i)

            # save preprocessed dataframe to pickle
            train.to_pickle(self.config.pickled_train_path)
            test.to_pickle(self.config.pickled_test_path)
            print("Succeeded in saving preprocessed dataframe to pickle")

        # create dataset
        Dataset = namedtuple("Dataset", ["train", "test"])
        dataset = Dataset(train, test)
        return dataset
