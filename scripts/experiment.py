from .config import Config
from .data import DatasetCreator
from .dataproc import DataProcessor
from .predict import Prediction
from .submission import Submission


class Experiment:
    def run(self):
        config = Config()
        print("Target:", config.target_name)

        creator = DatasetCreator(config)
        dataset = creator.run()
        del creator

        processor = DataProcessor(config)
        X_train, y_train, X_test = processor.run(dataset)
        del processor, dataset

        predict = Prediction(config)
        y_pred = predict.run(X_train, y_train, X_test)
        del predict, X_train, y_train, X_test

        submission = Submission(config)
        submission.save(y_pred)
        del config, submission

        return y_pred
