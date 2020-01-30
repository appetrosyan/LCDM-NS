import numpy
import pypolychord
import pypolychord.settings
import pypolychord.priors
import anesthetic
import matplotlib.pyplot as plt



def gaussian_likelihood(theta, mu, sig):
    # _, norm = numpy.linalg.slogdet(2 * numpy.pi * sigma)/2
    """ Simple Gaussian Likelihood. """
    invSig = numpy.linalg.inv(sig)
    norm = numpy.linalg.slogdet(2*numpy.pi*sig)[1]/2
    logL = - norm - (theta - mu) @ invSig @ (theta - mu) / 2

    #logL -= stuff that is related to the beta parameter
    return logL, []


def simple_prior(hypercube):
    """ Uniform prior from [-1,1]. """
    return pypolychord.priors.UniformPrior(-20, 20)(hypercube)


def uniform_prior_with_ranges(cube: numpy.ndarray, scale: numpy.ndarray):
    """
    A rescaling function that maps a point in pseudoparameter space into the actual parameter.
    Parameters
    ----------
    hypercube : point representing the parameters in a [0,1] hypercube

    scale : list of lists, representing the bounds on each parameter


    Returns
    -------
    out : scaled parameter values; theta

    """
    if not isinstance(cube, numpy.ndarray):
        cube = numpy.array(cube)
    if not isinstance(scale, numpy.ndarray):
        scale = numpy.array(scale)
    out = scale[:, 0] + cube[:]*(scale[:, 1] - scale[:, 0])
    return out






def exec_polychord(root_name: str, m, s, likelihood,
                   renew_plots=True, prior=simple_prior, nLive=200, fig=None, ax=None):
    nDerived = 0

    nDims = m.size

    settings = pypolychord.settings.PolyChordSettings(nDims, nDerived)
    settings.file_root = root_name
    settings.nlive = nLive
    settings.do_clustering = True
    settings.read_resume = False
    settings.feedback = 0
    # likelihood = lambda x: gaussian_likelihood(x, m, s)
    output = pypolychord.run_polychord(likelihood, nDims, nDerived,
                                       settings, prior)
    samples = anesthetic.NestedSamples(root='./chains/' +
                                       settings.file_root)
    # if fig is None and ax is None:
    if fig is None or renew_plots:
        fig, ax = samples.plot_2d([0, 1])
    else:
        samples.plot_2d(ax)


def find_samples(root_name: str, m, s, likelihood,
                 renew_plots=True, prior=simple_prior, nLive=200, fig=None, ax=None):
    try:
        return anesthetic.NestedSamples('./chains/'+root_name)
    except FileNotFoundError:
        nDerived = 0

        nDims = m.size

        settings = pypolychord.settings.PolyChordSettings(nDims, nDerived)
        settings.file_root = root_name
        settings.nlive = nLive
        settings.do_clustering = True
        settings.read_resume = False
        settings.feedback = 0
        # likelihood = lambda x: gaussian_likelihood(x, m, s)
        output = pypolychord.run_polychord(likelihood, nDims, nDerived,
                                           settings, prior)
        return anesthetic.NestedSamples(root='./chains/' +
                                        settings.file_root)



# Simple run
# ----------------------------------------------------------------------
# params = numpy.array([0, 0])
# p_cov = numpy.array([[1, 0.5],
#                      [0.5, 1]])
# fig = None
# ax = None
# exec_polychord('gaussian', params, p_cov,
#                lambda x: gaussian_likelihood(x, params, p_cov))
# exec_polychord('gaussian_2', params+0.5, p_cov,
#                lambda x: gaussian_likelihood(x, params + 0.5, p_cov),
#                renew_plots=False)
# plt.show()


# More params.
# ----------------------------------------------------------------------
# b_params = numpy.array([0.5, 0.5, 0.5, 0.5])
# b_cov = numpy.array([[1, 0.9, 0.7, 0.8],
#                      [0.9, 1, 0.6, 0.8],
#                      [0.7, 0.6, 1, 0],
#                      [0.8, 0.8, 0.8, 1]])
# exec_polychord('gaussian_4d', b_params, b_cov,
#                lambda x: gaussian_likelihood(x, b_params, b_cov))
# plt.show()

# Way more params
# ----------------------------------------------------------------------
planck_ranges = numpy.array(
            [[0.019, 0.025],
             [0.095, 0.145],
             [1.03, 1.05],
             [0.01, 0.4],
             [2.5, 3.7],
             [0.885, 1.04],
             [0.9, 1.1],
             [0, 200],
             [0, 1],
             [0, 10],
             [0, 400],
             [0, 400],
             [0, 400],
             [0, 400],
             [0, 10],
             [0, 50],
             [0, 50],
             [0, 100],
             [0, 400],
             [0, 10],
             [0, 10],
             [0, 10],
             [0, 10],
             [0, 10],
             [0, 10],
             [0, 3],
             [0, 3]])


samples = anesthetic.NestedSamples(root='./data.1908.09139/lcdm/chains/planck')
fig, ax = samples.plot_2d(['logA', 'ns'])
# plt.show()



params = samples.columns[:27]
Sig = samples[params].cov().values
mu = samples[params].mean().values
nDims = len(mu)

# Run of the original

args = {
    'root_name': 'planck',
    'm': mu,
    's': Sig,
    'likelihood': lambda x: gaussian_likelihood(x, mu, Sig),
    # 'renew_plots': True,
    'renew_plots': False,
    'nLive': 20,
    'prior': lambda x: uniform_prior_with_ranges(x, planck_ranges),
    'ax': ax
}
exec_polychord(**args)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Comparison of different runs of PolyChord on $\Lambda$CDM data from Planck.')
# newSamples = anesthetic.NestedSamples(root='./chains/planck') 
# newSamples.plot_2d(ax)

# plt.show()
# fig = plt.figure()
fig.savefig('./comparison.pdf')
