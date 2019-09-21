import inspect

import numpy as np
import pandas as pd

from .config import Config


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        if feature_names in globals():
            return globals()[feature_names](**kwargs)
        else:
            raise ValueError("No feature defined named with {}".format(feature_names))

    def feature_list(self):
        lst = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                Config,
                FeatureFactory,
                Feature,
                BasicFeature,
                NullFeature,
                NullPairFeature,
                LabeledFeature,
            ]:
                lst.append(obj.__name__)
        return lst


class Feature:

    fill_value = 0

    def __init___(self, **kwargs):
        self.name = str(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def run(self, df):
        values = self.extract(df)
        values = values.fillna(self.fill_value)
        return values

    def extract(self, df):
        raise NotImplementedError

    @staticmethod
    def _get_converted_multi_columns(values, head_name=None):
        col_names = [col[0] + "_" + col[1] for col in values.columns.values]
        if head_name:
            col_names = [head_name + "_" + col for col in col_names]
        return col_names


class BasicFeature(Feature):

    columns = None
    dummy = False
    prefix = None
    prefix_sep = None

    def extract(self, df):
        values = df[self.columns]
        if self.dummy:
            values = pd.get_dummies(
                values, prefix=self.prefix, prefix_sep=self.prefix_sep
            )
        return values


class NullFeature(Feature):

    columns = None

    def extract(self, df):
        cols = [col + "_null" for col in self.columns]
        values = df[self.columns].isnull() * 1
        values.columns = cols
        return values


class NullPairFeature(Feature):

    columns = None
    name = None

    def extract(self, df):
        def func(x):
            if np.isnan(x[0]):
                if np.isnan(x[1]):
                    return 0
                else:
                    return 1
            else:
                if np.isnan(x[1]):
                    return 2
                else:
                    return 3

        values = df[self.columns].apply(func, axis=1)
        values.name = self.name
        return values


class LabeledFeature(Feature):

    columns = None
    start = None
    end = None
    step = None

    def extract(self, df):
        def func(x):
            from_value = self.start
            label = 0
            while from_value < self.end:
                if from_value <= x < from_value + self.step:
                    return label
                from_value += self.step
                label += 1

        values = df[self.columns].map(func)
        values.name = self.columns + "_labeled"
        return values


class TransactionMonth(Feature):
    def extract(self, df):
        values = df["TransactionMonth"].map(lambda x: 1 if x == 1 else 0)
        return values


class TransactionHour(BasicFeature):

    columns = "TransactionHour"
    dummy = True
    prefix = "hour"
    prefix_sep = ""


class TransactionAmt(BasicFeature):

    columns = "TransactionAmt"


class ProductCD(BasicFeature):

    columns = "ProductCD"
    dummy = True
    prefix = "Product"
    prefix_sep = ""


class CardInfo(BasicFeature):

    columns = [f"card{n}" for n in range(1, 7) if n != 4]


class CardType(BasicFeature):

    columns = "card4"
    dummy = True


class Address(BasicFeature):

    columns = ["addr1", "addr2"]


class Distance(BasicFeature):

    columns = ["dist1", "dist2"]


class P_Emaildomain(BasicFeature):

    columns = "P_emaildomain"
    dummy = True
    prefix = "P"
    prefix_sep = "_"


class R_emaildomain(BasicFeature):

    columns = "R_emaildomain"
    dummy = True
    prefix = "R"
    prefix_sep = "_"


class Counting(BasicFeature):

    columns = [f"C{n}" for n in range(1, 15)]


class Timedelta(BasicFeature):

    columns = [f"D{n}" for n in range(1, 15)]


class D15(Feature):
    def extract(self, df):
        values = df["D15"].map(lambda x: 1 if x == 0 else 0)
        return values


class Match(BasicFeature):

    columns = [f"M{n}" for n in range(1, 10) if n != 4]


class M4(BasicFeature):

    columns = "M4"
    dummy = True
    prefix = "M4"
    prefix_sep = "_"


class Vesta(BasicFeature):

    columns = [f"V{n}" for n in range(1, 340)]


class Identity(BasicFeature):

    columns = ["id_" + f"{n}".zfill(2) for n in range(1, 39) if n != 30]
    columns += ["id_33_0", "id_33_1"]


class OSType(BasicFeature):

    columns = "id_30"
    dummy = True


class DeviceType(BasicFeature):

    columns = "DeviceType"


class DeviceInfo(BasicFeature):

    columns = "DeviceInfo"


class NullCardInfo(NullFeature):

    columns = [f"card{n}" for n in range(2, 7)]


class NullAddress(NullFeature):

    columns = ["addr1"]


class NullDistance(NullFeature):

    columns = ["dist1", "dist2"]


class NullEmaildomain(NullFeature):

    columns = ["P_emaildomain", "R_emaildomain"]


class NullTimedelta(NullFeature):

    columns = [f"D{n}" for n in range(1, 16)]


class NullMatch(NullFeature):

    columns = [f"M{n}" for n in range(1, 10)]


class NullVesta(NullFeature):

    columns = [f"V{n}" for n in range(1, 95)] + [f"V{n}" for n in range(138, 340)]


class NullIdentity(NullFeature):

    columns = ["id_" + f"{n}".zfill(2) for n in range(1, 39)] + ["id_33_0", "id_33_1"]


class NullDeviceType(NullFeature):

    columns = ["DeviceType"]


class NullDeviceInfo(NullFeature):

    columns = ["DeviceInfo"]


class NullPairDistance(NullPairFeature):

    columns = ["dist1", "dist2"]
    name = "dist_pair"


class NullPairEmaildomain(NullPairFeature):

    columns = ["P_emaildomain", "R_emaildomain"]
    name = "emaildomain_pair"


class LabeledCard1(LabeledFeature):

    columns = "card1"
    start = 1000
    end = 19000
    step = 1500


class LabeledCard2(LabeledFeature):

    columns = "card2"
    start = 100
    end = 600
    step = 50


class LabeledCard3(LabeledFeature):

    columns = "card3"
    start = 100
    end = 250
    step = 30


class LabeledCard5(LabeledFeature):

    columns = "card5"
    start = 100
    end = 250
    step = 30
