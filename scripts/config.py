import os

from .param_space import IntParamSpace, UniformParamSpace, LogUniformParamSpace


class Config:
    def __init__(self, nrows=None):
        # raw data path
        self.data_dir = "../data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_t_path = os.path.join(self.data_dir, "train_transaction.csv")
        self.test_t_path = os.path.join(self.data_dir, "test_transaction.csv")
        self.train_i_path = os.path.join(self.data_dir, "train_identity.csv")
        self.test_i_path = os.path.join(self.data_dir, "test_identity.csv")
        self.sample_submission_path = os.path.join(
            self.data_dir, "sample_submission.csv"
        )

        # pickled files path
        self.pickle_dir = "../pickle"
        self.pickled_data_dir = os.path.join(self.pickle_dir, "data")
        self.pickled_feature_dir = os.path.join(self.pickle_dir, "feature")
        pickled_feature_train_dir = os.path.join(self.pickled_feature_dir, "train")
        pickled_feature_test_dir = os.path.join(self.pickled_feature_dir, "test")
        self.pickled_model_dir = os.path.join(self.pickle_dir, "model")
        os.makedirs(self.pickle_dir, exist_ok=True)
        os.makedirs(self.pickled_data_dir, exist_ok=True)
        os.makedirs(self.pickled_feature_dir, exist_ok=True)
        os.makedirs(pickled_feature_train_dir, exist_ok=True)
        os.makedirs(pickled_feature_test_dir, exist_ok=True)
        os.makedirs(self.pickled_model_dir, exist_ok=True)
        self.pickled_train_path = os.path.join(self.pickled_data_dir, "train.pkl")
        self.pickled_test_path = os.path.join(self.pickled_data_dir, "test.pkl")

        # submission file path
        results_dir = "../results"
        os.makedirs(results_dir, exist_ok=True)
        self.submission_path = os.path.join(results_dir, "submission.csv")

        # used feature names
        self.feature_names = [
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "CardInfo",
            "Address",
            "Distance",
            "Emaildomain",
            "Counting",
            "Timedelta",
            "Match",
            "Vesta",
            "Identity",
            "DeviceType",
            "DeviceInfo",
            "CardInfoNull",
            "AddressNull",
            "DistanceNull",
            "EmaildomainNull",
            "TimedeltaNull",
            "MatchNull",
            "VestaNull",
            "IdentityNull",
            "DeviceTypeNull",
            "DeviceInfoNull",
        ]

        # target name
        self.target_name = "isFraud"

        # whether execute parameter tuning
        self.tuning = True

        # number of trials
        self.n_trials = 1

        # number of splits
        self.n_splits = 5

        # whether save predicted data
        self.save = False

        # study name and storage path of parameters for the best model
        database_dir = "../database"
        os.makedirs(database_dir, exist_ok=True)
        self.study_name = "lgb_2"
        self.storage_path = os.path.join(database_dir, f"{self.study_name}.db")

        # static parameters for model
        self.fixed_params = {
            "boosting_type": "gbdt",
            "max_depth": 20,
            "learning_rate": 1e-1,
            "n_estimators": 100000,
            "reg_alpha": 0.0,
            # "metric": "binary",
        }

        # parameter space for searching with optuna
        self.param_space = {
            "num_leaves": IntParamSpace("num_leaves", 2, 100),
            "subsample": UniformParamSpace("subsample", 0.5, 1.0),
            "subsample_freq": IntParamSpace("subsample_freq", 1, 20),
            "colsample_bytree": LogUniformParamSpace("colsample_bytree", 1e-2, 1e-1),
            # "min_child_weight": LogUniformParamSpace("min_child_weight", 1e-3, 1e1),
            # "min_child_samples": IntParamSpace("min_child_samples", 1, 50),
            # "reg_lambda": LogUniformParamSpace("reg_lambda", 1e-1, 1e4),
        }

        # random seed
        self.seed = 42
