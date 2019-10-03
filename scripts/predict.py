import os

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold

from .model import LGBMClassifier
from .tune import ParameterTuning


class Prediction:
    def __init__(self, config):
        self.config = config
        self.tuning = config.tuning
        self.n_splits = config.n_splits
        self.seed = config.seed
        self.save = config.save
        self.target_name = config.target_name
        self.pickled_model_dir = config.pickled_model_dir
        self.pickled_feature_dir = config.pickled_feature_dir

    def run(self, X_train, y_train, X_test):
        pt = ParameterTuning(self.config)
        if self.tuning is True:
            print("Started tuning")
            params = pt.run(X_train, y_train)
        else:
            params = pt.get_best_params()
            if params is None:
                params = {}

        model = LGBMClassifier(**params)
        folds = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        score = np.zeros(self.n_splits)
        y_pred = np.zeros((self.n_splits, X_test.shape[0]))
        for n, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            print(f"Fold {n+1}/{self.n_splits}")
            X_train_ = X_train.iloc[train_idx]
            y_train_ = y_train.iloc[train_idx]
            X_valid_ = X_train.iloc[valid_idx]
            y_valid_ = y_train.iloc[valid_idx]
            model.fit(
                X_train_,
                y_train_,
                eval_set=[(X_train_, y_train_), (X_valid_, y_valid_)],
                eval_names=["train", "valid"],
                early_stopping_rounds=50,
                verbose=5000,
            )
            self._save_trained_model(model, n + 1)
            y_pred_ = model.predict(X_valid_)
            score[n] = model.calculate_score(y_valid_, y_pred_)
            y_pred[n] = model.predict(X_test)
            del X_train_, y_train_, X_valid_, y_valid_, y_pred_
        print(f"Score: {score.mean()}")
        y_pred = y_pred.mean(axis=0)

        if self.save is True:
            self._save_predicted_feature(y_pred, X_test.index)

        return y_pred

    def _save_trained_model(self, model, n):
        with open(os.path.join(self.pickled_model_dir, f"model_{n}.pkl"), "wb") as f:
            pickle.dump(model, f)
            print(f"Succeeded in saving model_{n} to pickle")

    def _save_predicted_feature(self, y_pred, index):
        predicted_feature = pd.DataFrame(y_pred, index=index, columns=self.target_name)
        path = os.path.join(self.pickled_feature_dir, "test", f"{self.target_name}.pkl")
        predicted_feature.to_pickle(path)
        print(f"Succeeded in saving {self.target_name} for test to pickle")
