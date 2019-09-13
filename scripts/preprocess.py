import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .utils import reduce_mem_usage


class Preprocessor:
    def __init__(self, config):
        self.target_name = config.target_name

    def run(self, train_t, test_t, train_i, test_i):
        print("Started preprocessing")
        test_t[self.target_name] = 0

        train_t, test_t = self.preprocess_transaction(train_t, test_t)
        train_i, test_i = self.preprocess_identity(train_i, test_i)

        train = train_t.join(train_i)
        test = test_t.join(test_i)

        print("Finished preprocessing")
        return train, test

    def preprocess_transaction(self, train, test):
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        train = self._to_month(train)
        test = self._to_month(test)

        train = self._logarithmization(train)
        test = self._logarithmization(test)

        train = self._rename_emaildomain(train)
        test = self._rename_emaildomain(test)

        cols = ["card4", "card6"]
        cols += ["ProductCD"]
        cols += ["M4"]
        cols += ["P_emaildomain", "R_emaildomain"]
        cols += ["addr1", "addr2"]
        cols += [f"C{n}" for n in range(1, 15)]
        train, test = self._convert_string_to_count(train, test, cols)

        cols = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
        train = self._convert_bool_to_int(train, cols)
        test = self._convert_bool_to_int(test, cols)

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        train = train.set_index("TransactionID")
        test = test.set_index("TransactionID")

        return train, test

    def preprocess_identity(self, train, test):
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        train = self._rename_OS(train)
        test = self._rename_OS(test)

        train = self._rename_browser(train)
        test = self._rename_browser(test)

        cols = ["id_30", "id_31", "DeviceInfo"]
        train, test = self._convert_string_to_count(train, test, cols)

        cols = ["id_35", "id_36", "id_37", "id_38"]
        train = self._convert_bool_to_int(train, cols)
        test = self._convert_bool_to_int(test, cols)

        train = self._convert_string_to_int(train)
        test = self._convert_string_to_int(test)

        train, test = self._convert_string_to_label(train, test)

        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        train = train.set_index("TransactionID")
        test = test.set_index("TransactionID")

        return train, test

    @staticmethod
    def _to_month(df):
        sec2day = 60 * 60 * 24
        year2day = 365.25
        day2month = year2day / 12

        def map(x):
            x /= sec2day
            if x > year2day:
                x -= year2day
            month = 1
            while month < 13:
                if (month - 1) * day2month <= x < month * day2month:
                    x = month
                    break
                month += 1
            if x > 12:
                x = 12
            return x

        df["TransactionDT"] = df["TransactionDT"].map(map)
        return df

    @staticmethod
    def _logarithmization(df):
        df["TransactionAmt"] = np.log10(df["TransactionAmt"])
        return df

    @staticmethod
    def _fill_distance(train, test, col):
        _df = pd.concat([train[["addr1", col]], test[["addr1", col]]]).dropna()
        _df = _df.drop_duplicates("addr1").drop_duplicates(col)
        addr2dist = {addr: dist for addr, dist in zip(_df["addr1"], _df[col])}
        train_idx = train[col].isnull()
        test_idx = test[col].isnull()
        train.loc[train_idx, col] = train.loc[train_idx, "addr1"].map(addr2dist)
        test.loc[test_idx, col] = test.loc[test_idx, "addr1"].map(addr2dist)
        return train, test

    @staticmethod
    def _rename_emaildomain(df):
        cols = ["P_emaildomain", "R_emaildomain"]
        for col in cols:
            df[col] = df[col].fillna("null")
            df[col] = df[col].map(lambda x: x.split(".")[0])
            df[col] = df[col].replace("null", np.nan)
        return df

    @staticmethod
    def _rename_OS(df):
        def rename(x):
            for name in OS_names:
                if name in str(x):
                    x = name
            return x

        OS_names = ["android", "ios", "mac", "windows", "linux", "func"]
        df["id_30"] = df["id_30"].fillna("null").map(lambda x: x.lower()).map(rename)
        df["id_30"] = df["id_30"].replace("null", np.nan)
        return df

    @staticmethod
    def _rename_browser(df):
        def rename(x):
            for name in browser_names:
                if name in str(x):
                    x = name
            return x

        browser_names = [
            "samsung",
            "safari",
            "chrome",
            "edge",
            "firefox",
            "ie",
            "webview",
            "generic",
            "opera",
            "android",
            "google",
        ]
        df["id_31"] = df["id_31"].fillna("null").map(lambda x: x.lower()).map(rename)
        df["id_31"] = df["id_31"].replace("null", np.nan)
        return df

    @staticmethod
    def _convert_string_to_count(train, test, cols):
        for col in cols:
            _df = pd.concat([train[[col]], test[[col]]])
            count_dict = _df[col].value_counts().to_dict()
            train[col] = train[col].map(count_dict)
            test[col] = test[col].map(count_dict)
        return train, test

    @staticmethod
    def _convert_bool_to_int(df, cols):
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
        df["DeviceType"] = df["DeviceType"].map({"desktop": 1, "mobile": 0})
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
