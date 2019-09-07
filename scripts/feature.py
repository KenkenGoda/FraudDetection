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


class TransactionDT(BasicFeature):

    columns = "TransactionDT"


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

    columns = [f"D{n}" for n in range(1, 16)]


class Match(BasicFeature):

    columns = [f"M{n}" for n in range(1, 10)]


class Vesta(BasicFeature):

    columns = [f"V{n}" for n in range(1, 340)]


class Identity(BasicFeature):

    columns = ["id_" + f"{n}".zfill(2) for n in range(1, 39)]
    columns += ["id_33_0", "id_33_1"]


class DeviceType(BasicFeature):

    columns = "DeviceType"


class DeviceInfo(BasicFeature):

    columns = "DeviceInfo"