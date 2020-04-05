from polychord_model import Model


class OffsetModel(Model):
    defaul_file_root = 'OffsetModel'

    def __init__(self, baseModel, offset, **kwargs):
        self.model = baseModel

    def loglikelihood(self, theta):
        return self.model.loglikelihood(theta-offset)
