import matplotlib.pyplot as plt
from polychord_model import Model
from power_posterior import PowerPosteriorPrior
from uniform import BoxUniformModel
from resizeableUniform import ResizeableUniformPrior
from pypolychord.settings import PolyChordSettings
from numpy import array, concatenate, lcm, log
from random import random, seed
import fractions


class StochasticMixtureModel(Model):
    default_file_root = 'mixture'

    def __init__(self, models, settings=None, file_root=default_file_root):
        """
        Constructor for the general mixture model, given the models in args.
        """
        self.models = models

        if not allElementsIdentical(list(map(lambda x: x.nDims, models))):
            raise ValueError("not all models have the same nDims.")
        self.nDims = max(map(lambda x: x.eff_nDims(), models))

        if not allElementsIdentical(list(map(lambda x: x.nDerived, models))):
            raise ValueError("not all models have the same nDerived.")
        else:
            self.nDerived = models[0].nDerived

        self._eff_nDims = self.nDims + len(self.models)

        if settings is None:
            self.settings = PolyChordSettings(self.eff_nDims(), self.nDerived)
        else:
            self.settings = settings
        self.settings.file_root = file_root

    def eff_nDims(self):
        return self._eff_nDims

    def _unpack(self, theta):
        physical_params = theta[:self.nDims]
        choice_probabilities = theta[self.nDims:-1]
        index = theta[-1:].item()
        return physical_params, choice_probabilities, index

    def loglikelihood(self, theta):
        t, _, m = self._unpack(theta)
        _current_model = self.models[int(m)]
        _nDims = _current_model.eff_nDims()
        logl, phi = _current_model.loglikelihood(t[:_nDims])
        return logl, phi

    def prior_inversion_function(self, hypercube):
        t, b, _ = self._unpack(hypercube)
        norm = b.sum() if b.sum() != 0 else 1
        ps = b/norm
        index = 0
        # h = abs(hash(tuple(t)))/(2**63-1)
        h = hash(tuple(t))
        seed(h)
        r = random()
        for p in ps:
            if r > p:
                # if h > p:
                break
            index += 1
        _nDims = self.models[index].eff_nDims()
        cube, cube_ = t[:_nDims], t[_nDims:]
        theta = self.models[index].prior_inversion_function(cube)
        return concatenate([theta, cube_, b, [index]])

    def _test_prior(self):
        for m in self.models:
            m._test_prior()
        super()._test_prior()

    def _test_loglike(self):
        for m in self.models:
            m._test_loglike()
        super()._test_loglike()


def allElementsIdentical(lst):
    return not lst or lst.count(lst[0]) == len(lst)
