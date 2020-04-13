from numpy import pad

from polychord_model import Model


class OffsetModel(Model):
    default_file_root = 'OffsetModel'

    def __str__(self):
        return f'{self.model.__str__()}\nOffset by {self.offset}'

    def __repr__(self):
        return f'{self.model.__repr__()} Offset by {self.offset}'

    def __init__(self, base_model, offset, file_root=default_file_root, **kwargs):
        self.model = base_model
        self.offset = offset
        if not self.offset.size == self.model.dimensionality:
            self.offset = pad(self.offset, (0, abs(
                self.model.dimensionality - self.offset.size)))
        super().__init__(base_model.dimensionality,
                         base_model.num_derived, file_root=file_root, **kwargs)

    def log_likelihood(self, theta):
        return self.model.log_likelihood(theta - self.offset)

    def quantile(self, *args):
        return self.model.quantile(*args)

    @property
    def dimensionality(self):
        return self.model.dimensionality
