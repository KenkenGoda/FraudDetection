import os

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from .model import LGBMClassifier


class ParameterTuning:
    def __init__(self, config):
        self.study_name = config.study_name
        self.storage_path = config.storage_path
        self.n_trials = config.n_trials
        self.n_splits = config.n_splits
        self.seed = config.seed
        self.fixed_params = config.fixed_params
        self.param_space = config.param_space
        self.target_name = config.target_name

    def run(self, X, y):
        def objective(trial):
            params = {key: space(trial) for key, space in self.param_space.items()}
            params.update(self.fixed_params)
            model = LGBMClassifier(**params)
            folds = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
            score = np.zeros(self.n_splits)
            for i, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
                print(f"Fold {i+1}/{self.n_splits} on tuning")
                X_train_ = X.iloc[train_idx]
                y_train_ = y.iloc[train_idx]
                X_valid_ = X.iloc[valid_idx]
                y_valid_ = y.iloc[valid_idx]
                model.fit(
                    X_train_,
                    y_train_,
                    eval_set=(X_valid_, y_valid_),
                    early_stopping_rounds=3,
                    verbose=500,
                )
                y_pred_ = model.predict(X_valid_)
                score[i] = model.calculate_score(y_valid_, y_pred_)
                del X_train_, y_train_, X_valid_, y_valid_, y_pred_
            score = score.mean()
            return score

        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.storage_path}",
            load_if_exists=True,
        )

        # search the best parameters with optuna
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)

        # get the best parameters
        best_params = study.best_params
        best_params.update(self.fixed_params)
        return best_params

    def get_best_params(self):
        if os.path.isfile(self.storage_path):
            study = optuna.create_study(
                study_name=self.study_name,
                storage=f"sqlite:///{self.storage_path}",
                load_if_exists=True,
            )
            print("load the best parameters")
            params = study.best_params
            params.update(self.fixed_params)
            print(params)
            return params
        else:
            print("Not found the best parameters")
            return None
