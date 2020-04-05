from pypolychord.settings import PolyChordSettings
import matplotlib.pyplot as plt
import pypolychord as ppc
from pypolychord.priors import UniformPrior

def quantile(cube):
    return UniformPrior(-10, 10)(cube)


def lnL(theta):
    return theta**2


settings = PolyChordSettings(2, 0)
ppc.run_polychord(lnL, 2, 0, settings, quantile)
