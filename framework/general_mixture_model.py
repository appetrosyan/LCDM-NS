import fractions
import matplotlib.pyplot as plt
from numpy import array, concatenate, lcm, log
from pypolychord.settings import PolyChordSettings
from random import random, seed

from polychord_model import Model


def _are_all_elements_identical(lst):
    return not lst or lst.count(lst[0]) == len(lst)


class AbstractMixtureModel(Model):
    default_file_root = 'MixtureModel'

    def __init__(self, models, file_root=default_file_root, **kwargs):
        self.models = models
        if not _are_all_elements_identical([x.nDims for x in models]):
            raise ValueError(
                "Models in mixture have different dimensionality.")
        self.nDims = max([x.dimensionality for x in models])
        if not _are_all_elements_identical([x.num_derived for x in models]):
            raise ValueError(
                "Models in mixture have different derived parameters.")
        self.nDerived = models[0].num_derived
        super().__init__(self.dimensionality, self.num_derived, file_root, **kwargs)

    def test_quantile(self):
        for m in self.models:
            m.test_quantile()
        super().test_quantile()

    def test_log_like(self):
        for m in self.models:
            m.test_log_like()
        super().test_log_like()

    @property
    def num_derived(self):
        return self.nDerived


class StochasticMixtureModel(AbstractMixtureModel):
    default_file_root = 'StochasticMixture'

    def __init__(self, models, settings=None, file_root=default_file_root):
        super().__init__(models, file_root=file_root, settings=settings)

    @property
    def dimensionality(self):
        return self.nDims + len(self.models)

    def _unpack(self, theta):
        physical_params = theta[:self.nDims]
        choice_probabilities = theta[self.nDims:-1]
        index = theta[-1:].item()
        return physical_params, choice_probabilities, index

    def log_likelihood(self, theta):
        t, _, m = self._unpack(theta)
        _current_model = self.models[int(m)]
        _nDims = _current_model.dimensionality
        logl, phi = _current_model.log_likelihood(t[:_nDims])
        return logl, phi

    def quantile(self, hypercube):
        """This is where black magic happens!"""
        t, b, _ = self._unpack(hypercube)
        norm = b.sum() if b.sum() != 0 else 1
        ps = b / norm
        index = 0
        h = hash(tuple(t))
        seed(h)
        r = random()
        for p in ps:
            if r > p:
                break
            index += 1
        _nDims = self.models[index].dimensionality
        cube, cube_ = t[:_nDims], t[_nDims:]
        theta = self.models[index].quantile(cube)
        return concatenate([theta, cube_, b, [index]])
