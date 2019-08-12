class ParamSpace(object):

    suggest_method = None

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, trial):
        return getattr(trial, self.suggest_method)(self.name, *self.args, **self.kwargs)


class FixSpace(ParamSpace):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, trial):
        return self.value


FixedSpace = FixSpace


class IntParamSpace(ParamSpace):

    suggest_method = "suggest_int"

    def __init__(self, name, low, high):
        super().__init__(name, low, high)


class UniformParamSpace(ParamSpace):

    suggest_method = "suggest_uniform"

    def __init__(self, name, low, high):
        super().__init__(name, low, high)


class CategoricalParamSpace(ParamSpace):

    suggest_method = "suggest_categorical"

    def __init__(self, name, choices):
        super().__init__(name, choices)


class DiscreteUniformParamSpace(ParamSpace):

    suggest_method = "suggest_discrete_uniform"

    def __init__(self, name, low, high, q):
        super().__init__(name, low, high, q)


class LogUniformParamSpace(ParamSpace):

    suggest_method = "suggest_loguniform"

    def __init__(self, name, low, high):
        super().__init__(name, low, high)
