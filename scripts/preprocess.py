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

        train = self._preprocess_TransactionDT(train)
        test = self._preprocess_TransactionDT(test)

        train, test = self._preprocess_TransactionAmt(train, test)

        train = self._add_card_count(train)
        test = self._add_card_count(test)

        train, test = self._fill_distance(train, test, "dist1")
        train, test = self._fill_distance(train, test, "dist2")

        train = self._rename_emaildomain(train)
        test = self._rename_emaildomain(test)

        train = self._make_uid(train)
        test = self._make_uid(test)

        cols = ["TransactionMonth", "TransactionDayOfWeek", "TransactionHour"]
        cols += ["ProductCD"]
        cols += [f"card{n}" for n in range(1, 7)]
        cols += ["addr1", "addr2"]
        cols += ["dist1", "dist2"]
        cols += ["P_emaildomain", "R_emaildomain"]
        cols += [f"C{n}" for n in range(1, 15)]
        cols += [f"D{n}" for n in range(1, 16) if n != 9]
        cols += [f"V{n}" for n in range(1, 340)]
        cols += ["uid", "uid1", "uid2", "uid3", "uid4"]
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

        train = self._make_id(train)
        test = self._make_id(test)

        train = self._preprocess_DeviceInfo(train)
        test = self._preprocess_DeviceInfo(test)

        cols = ["id_30", "id_31", "DeviceInfo", "DeviceInfo1", "DeviceInfo2"]
        cols += ["id", "id1", "id2"]
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
    def _preprocess_TransactionDT(df):
        df["TransactionDT"] = pd.to_datetime(df["TransactionDT"], unit="s")
        df["TransactionMonth"] = df["TransactionDT"].dt.month
        df["TransactionDayOfWeek"] = df["TransactionDT"].dt.day_name()
        df["TransactionHour"] = df["TransactionDT"].dt.hour
        return df

    @staticmethod
    def _preprocess_TransactionAmt(train, test):
        amt_train = train["TransactionAmt"]
        amt_test = test["TransactionAmt"]
        train["TransactionAmtCheck"] = np.where(amt_train.isin(amt_test), 1, 0)
        test["TransactionAmtCheck"] = np.where(amt_test.isin(amt_train), 1, 0)
        train["TransactionAmtDecimal"] = (
            (amt_train - amt_train.astype(int)) * 1000
        ).astype(int)
        test["TransactionAmtDecimal"] = (
            (amt_test - amt_test.astype(int)) * 1000
        ).astype(int)
        train["TransactionAmt"] = np.log10(amt_train)
        test["TransactionAmt"] = np.log10(amt_test)
        return train, test

    @staticmethod
    def _add_card_count(df):
        cols = [f"card{n}" for n in range(1, 7)]
        cols += ["addr1"]
        cols += [f"C{n}" for n in range(1, 12)]
        _df = df.groupby(cols, as_index=False)["TransactionID"].count()
        _df = _df.rename(columns={"TransactionID": "cardCount"})
        df = df.merge(_df, how="left", on=cols)
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
            df[col] = df[col].replace("scranton", np.nan)
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
    def _make_uid(df):
        df["uid"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
        df["uid1"] = df["uid"] + "_" + df["card3"].astype(str)
        df["uid2"] = df["uid1"] + "_" + df["card5"].astype(str)
        df["uid3"] = (
            df["uid2"] + "_" + df["addr1"].astype(str) + "_" + df["addr2"].astype(str)
        )
        df["uid4"] = df["card4"].astype(str) + "_" + df["card6"].astype(str)
        return df

    @staticmethod
    def _make_id(df):
        df["id"] = df["id_35"].astype(str) + "_" + df["id_36"].astype(str)
        df["id1"] = df["id"] + "_" + df["id_37"].astype(str)
        df["id2"] = df["id1"] + "_" + df["id_38"].astype(str)
        return df

    @staticmethod
    def _preprocess_DeviceInfo(df):
        cols = ["DeviceInfo1", "DeviceInfo2"]
        df[cols] = df["DeviceInfo"].str.split("/", expand=True)[[0, 1]]
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

        df["id_33"] = df["id_33"].fillna("0x0")
        df["id_33_0"] = df["id_33"].apply(lambda x: x.split("x")[0]).astype(int)
        df["id_33_1"] = df["id_33"].apply(lambda x: x.split("x")[1]).astype(int)
        df["id_33_2"] = df["id_33_0"] * df["id_33_1"]
        df["id_33"] = np.where(df["id_33"] == "0x0", np.nan, df["id_33"])

        df["id_34"] = df["id_34"].fillna(":0")
        df["id_34"] = df["id_34"].map(lambda x: x.split(":")[1]).astype(np.int8)
        df["id_34"] = np.where(df["id_34"] == 0, np.nan, df["id_34"])
        df["id_34"] = np.where(df["id_34"] == -1, np.nan, df["id_34"])

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
