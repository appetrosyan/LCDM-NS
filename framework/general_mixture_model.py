import matplotlib.pyplot as plt
from polychord_model import Model
from uniform import BoxUniformModel
from pypolychord.settings import PolyChordSettings
from numpy import array
import fractions


class MixtureModel(Model):
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

        self._eff_nDims = self.nDims + \
            sum(map(lambda x: x.eff_nDims() - x.nDims, models)) + \
            len(self.models)

        if settings is None:
            self.settings = PolyChordSettings(self.eff_nDims(), self.nDerived)
        else:
            self.settings = settings
        self.settings.file_root = file_root

    def eff_nDims(self):
        return self._eff_nDims

    def loglikelihood(self, theta):
        t = theta[:self.nDims]  # All physical-like parameters,
        #  they can have repartitioning parameters in them
        b = theta[self.nDims:-1]  # Partition choice probabilities
        m = theta[-1:].item()     # Partition choice
        print(theta, t, b, m)
        rationals = list(map(lambda x: fractions.Fraction(x), b))
        print(rationals)
        fiduciary = hash(tuple(theta))
        return 0, []

    def prior_inversion_function(self, hypercube):
        pass


def allElementsIdentical(lst):
    return not lst or lst.count(lst[0]) == len(lst)


mdl1 = BoxUniformModel(
    (-2000, 2000), array([1, 2, 3]), array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
mdl2 = BoxUniformModel(
    (-2000, 2000), array([1, 2]), array([[1, 0], [0, 1]]))

# try:
#     mix = MixtureModel([mdl1, mdl2])
# except ValueError as e:
#     print(e)

mix = MixtureModel([mdl1, mdl1, mdl1])
