import os

import pandas as pd


class LocalFile:
    def __init__(self, config):
        self.pickled_train_path = config.pickled_train_path
        self.pickled_test_path = config.pickled_test_path
        self.train_identity_path = config.train_identity_path
        self.test_identity_path = config.test_identity_path
        self.train_transaction_path = config.train_transaction_path
        self.test_transaction_path = config.test_transaction_path
        self.sample_submission_path = config.sample_submission_path

    def get_train(self):
        if os.path.isfile(self.pickled_train_path):
            return pd.read_pickle(self.pickled_train_path)
        else:
            identity = self.get_train_identity()
            transaction = self.get_train_transaction()
            return transaction.join(identity)

    def get_test(self):
        if os.path.isfile(self.pickled_test_path):
            return pd.read_pickle(self.pickled_test_path)
        else:
            identity = self.get_test_identity()
            transaction = self.get_test_transaction()
            return transaction.join(identity)

    def get_train_identity(self):
        return pd.read_csv(self.train_identity_path).set_index('TransactionID')

    def get_test_identity(self):
        return pd.read_csv(self.test_identity_path).set_index('TransactionID')

    def get_train_transaction(self):
        return pd.read_csv(self.train_transaction_path).set_index('TransactionID')

    def get_test_transaction(self):
        return pd.read_csv(self.test_transaction_path).set_index('TransactionID')

    def get_submission(self):
        return pd.read_csv(self.sample_submission_path)
