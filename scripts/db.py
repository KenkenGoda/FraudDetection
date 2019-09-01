import pandas as pd


class LocalFile:
    def __init__(self, config):
        self.train_t_path = config.train_t_path
        self.test_t_path = config.test_t_path
        self.train_i_path = config.train_i_path
        self.test_i_path = config.test_i_path
        self.sample_submission_path = config.sample_submission_path

    def get_train_transaction(self):
        return pd.read_csv(self.train_t_path)

    def get_test_transaction(self):
        return pd.read_csv(self.test_t_path)

    def get_train_identity(self):
        return pd.read_csv(self.train_i_path)

    def get_test_identity(self):
        return pd.read_csv(self.test_i_path)

    def get_submission(self):
        return pd.read_csv(self.sample_submission_path)
