from .config import Config
from .data import DatasetCreator
from .dataproc import DataProcessor
from .predict import Prediction


class Experiment:
    def __init__(self, dataset=None, X_train=None, y_train=None, X_test=None):
        self.dataset_ = dataset
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_test_ = X_test

    def run(self):
        config = Config()
        print("Target:", config.target_name)

        if self.dataset_ is None:
            creator = DatasetCreator(config)
            self.dataset_ = creator.run()

        if self.X_train_ is None:
            processor = DataProcessor(config)
            self.X_train_, self.y_train_, self.X_test_ = processor.run(self.dataset_)

        predict = Prediction(config)
        y_pred = predict.run(self.X_train_, self.y_train_, self.X_test_)
        return y_pred
