import os

from collections import namedtuple

from .config import Config
from .db import LocalFile
from .preprocess import Preprocessor


class DatasetCreator:
    def __init__(self):
        self.config = Config()

    def run(self):
        # load files
        db = LocalFile(self.config)
        train = db.get_train()
        test = db.get_test()

        # preprocess data
        if not os.path.isfile(self.config.pickled_train_path):
            preprocessor = Preprocessor()
            train, test = preprocessor.run(train, test)

            # save preprocessed dataframe to pickle
            train.to_pickle(self.config.pickled_train_path)
            test.to_pickle(self.config.pickled_test_path)

        # create dataset
        Dataset = namedtuple("Dataset", ["train", "test"])
        dataset = Dataset(train, test)
        return dataset
