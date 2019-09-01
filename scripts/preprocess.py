import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .utils import reduce_mem_usage


class Preprocessor:
    def run(self, train, test):
        print("Start preprocessing")
        test["isFraud"] = 0

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        train, test = self._convert_string_to_count(train, test)

        train = self._convert_bool_to_int(train)
        test = self._convert_bool_to_int(test)

        train = self._convert_string_to_int(train)
        test = self._convert_string_to_int(test)

        train, test = self._convert_string_to_label(train, test)

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        print("Finish preprocessing")
        return train, test

    @staticmethod
    def _convert_string_to_count(train, test):
        cols = ["card4", "card6", "ProductCD", "M4"]
        for col in cols:
            _df = pd.concat([train[[col]], test[[col]]])
            count_dict = _df[col].value_counts().to_dict()
            train[col] = train[col].map(count_dict)
            test[col] = test[col].map(count_dict)
        return train, test

    @staticmethod
    def _convert_bool_to_int(df):
        cols = [
            "M1",
            "M2",
            "M3",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
        ]
        for col in cols:
            df[col] = df[col].map({"T": 1, "F": 0})
        return df

    @staticmethod
    def _convert_string_to_int(df):
        df["id_12"] = df["id_12"].map({"Found": 1, "NotFound": 0})
        df["id_15"] = df["id_15"].map({"New": 2, "Found": 1, "Unknown": 0})
        df["id_16"] = df["id_16"].map({"Found": 1, "NotFound": 0})
        df["id_23"] = df["id_23"].map(
            {
                "TRANSPARENT": 4,
                "IP_PROXY": 3,
                "IP_PROXY:ANONYMOUS": 2,
                "IP_PROXY:HIDDEN": 1,
            }
        )
        df["id_27"] = df["id_27"].map({"Found": 1, "NotFound": 0})
        df["id_28"] = df["id_28"].map({"New": 2, "Found": 1})
        df["id_29"] = df["id_29"].map({"Found": 1, "NotFound": 0})
        df["id_34"] = df["id_34"].fillna(":0")
        df["id_34"] = df["id_34"].apply(lambda x: x.split(":")[1]).astype(np.int8)
        df["id_34"] = np.where(df["id_34"] == 0, np.nan, df["id_34"])
        df["id_33"] = df["id_33"].fillna("0x0")
        df["id_33_0"] = df["id_33"].apply(lambda x: x.split("x")[0]).astype(int)
        df["id_33_1"] = df["id_33"].apply(lambda x: x.split("x")[1]).astype(int)
        df["id_33"] = np.where(df["id_33"] == "0x0", np.nan, df["id_33"])
        df["DeviceType"].map({"desktop": 1, "mobile": 0})
        return df

    @staticmethod
    def _convert_string_to_label(train, test):
        col = "id_33"
        train[col] = train[col].fillna("unseen_before_label")
        test[col] = test[col].fillna("unseen_before_label")
        le = LabelEncoder()
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        return train, test
