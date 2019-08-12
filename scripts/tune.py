import os

import numpy as np
import optuna
from sklearn.model_selection import KFold

from .model import LGBMRegressor


class ParameterTuning:
    def __init__(self, config):
        self.target_name = config.target_name
        self.study_name = config.study_name
        self.storage_path = config.storage_path
        self.fixed_params = config.fixed_params
        self.param_space = config.param_space
        self.seed = config.seed

    def run(self, X, y, n_trials=1, n_splits=2):
        def objective(trial):
            params = {key: space(trial) for key, space in self.param_space.items()}
            params.update(self.fixed_params)
            model = LGBMRegressor(**params)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            score = []
            for train_idx, valid_idx in kf.split(X):
                X_train_ = X.iloc[train_idx]
                y_train_ = y.iloc[train_idx][self.target_name]
                X_valid_ = X.iloc[valid_idx]
                y_valid_ = y.iloc[valid_idx]
                model.fit(
                    X_train_,
                    y_train_,
                    eval_set=(X_valid_, y_valid_[self.target_name]),
                    early_stopping_rounds=3,
                    verbose=False,
                )
                y_pred_ = model.predict(X_valid_)
                score.append(model.calculate_score(y_valid_, y_pred_))
            score = np.mean(score)
            return score

        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.storage_path}",
            load_if_exists=True,
        )

        # search the best parameters with optuna
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

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
