import inspect

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
    prefix = None

    def extract(self, df):
        values = df[self.columns]
        return values


class NullFeature(Feature):

    columns = None

    def extract(self, df):
        cols = [col + "_null" for col in self.columns]
        values = df[self.columns].isnull() * 1
        values.columns = cols
        return values


class LabeledFeature(Feature):

    columns = None
    start = None
    end = None
    step = None

    def extract(self, df):
        def func(x):
            from_value = self.start
            while from_value < self.end:
                if from_value <= x < from_value + self.step:
                    return int(from_value / self.step)
                from_value += self.step

        values = df[self.columns].map(func)
        return values


class TransactionDT(Feature):
    def extract(self, df):
        values = df["TransactionDT"].map(lambda x: 1 if x == 1 else 0)
        return values


class TransactionAmt(BasicFeature):

    columns = "TransactionAmt"


class ProductCD(BasicFeature):

    columns = "ProductCD"


class CardInfo(BasicFeature):

    columns = [f"card{n}" for n in range(1, 7)]


class Address(BasicFeature):

    columns = ["addr1", "addr2"]


class Distance(BasicFeature):

    columns = ["dist1", "dist2"]


class Emaildomain(BasicFeature):

    columns = ["P_emaildomain", "R_emaildomain"]


class Counting(BasicFeature):

    columns = [f"C{n}" for n in range(1, 15)]


class Timedelta(BasicFeature):

    columns = [f"D{n}" for n in range(1, 15)]


class D15(Feature):
    def extract(self, df):
        values = df["D15"].map(lambda x: 1 if x == 0 else 0)
        return values


class Match(BasicFeature):

    columns = [f"M{n}" for n in range(1, 10)]


class Vesta(BasicFeature):

    columns = [f"V{n}" for n in range(1, 340)]


class Identity(BasicFeature):

    columns = ["id_" + f"{n}".zfill(2) for n in range(1, 39)] + ["id_33_0", "id_33_1"]


class DeviceType(BasicFeature):

    columns = "DeviceType"


class DeviceInfo(BasicFeature):

    columns = "DeviceInfo"


class NullCardInfo(NullFeature):

    columns = [f"card{n}" for n in range(2, 7)]


class NullAddress(NullFeature):

    columns = ["addr1", "addr2"]


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


class LabeledCard1(LabeledFeature):

    columns = ["card1"]
    start = 1000
    end = 19000
    step = 1500


class LabeledCard2(LabeledFeature):

    columns = ["card2"]
    start = 100
    end = 600
    step = 50
