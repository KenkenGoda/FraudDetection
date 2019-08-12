import numpy as np
import pandas as pd


class Preprocessor:
    def run(self, train, test):
        train = self.preprocess_train(train)
        test = self.preprocess_test(test)
        return train, test

    def preprocess_train(self, train):
        return train

    def preprocess_test(self, test):
        return test

