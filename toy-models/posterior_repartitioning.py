from numpy import pi, log, sqrt, exp, log
from numpy import concatenate as concat
from scipy.special import erfinv, erf
import numpy
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import GaussianPrior
from anesthetic import NestedSamples
import matplotlib.pyplot as plt


# | Define a four-dimensional spherical gaussian likelihood,
# | width sigma=0.1, centered on the 0 with one derived parameter.
# | The derived parameter is the squared radius

nDims = 4
nDerived = 0
sigma = 0.1
mu = 0
nlive = 72
thetamin, thetamax = -1000, 1000
betamin, betamax = 0, 1
sigmaLL = ((thetamax - thetamin)/sqrt(2*pi))
# outputs = {}
# samples = {}
# colors = {}
normal_settings = PolyChordSettings(nDims, nDerived)
normal_settings.read_resume = False
normal_settings.feedback = 0
normal_settings.nlive = nlive
normal_settings.nDims = nDims
normal_settings.nDerived = nDerived

repart_settings = PolyChordSettings(nDims+1, nDerived)
repart_settings.read_resume = False
repart_settings.feedback = 0
repart_settings.nlive = nlive
repart_settings.nDims = nDims+1
repart_settings.nDerived = nDerived


def rank_nlike_calls(outs):
    for k in sorted(outputs, key=(lambda x: outputs[x].nlike)):
        print('{} {:.2E}'.format(k, outs[k].nlike))


def loglikelihood(theta):
    """ Simple Gaussian Likelihood. Caution it's not normalised"""
    return -(theta-mu) @ (theta-mu)/2/sigma**2, []


def uniform_prior(hypercube):
    return thetamin + hypercube*(thetamax-thetamin)


def gaussian_prior(hypercube):
    return GaussianPrior(mu, sigma)(hypercube)


def repartitioned_prior(hypercube):
    x = hypercube[:nDims]
    b = hypercube[-1]
    beta = betamin + (betamax-betamin)*b
    theta = power_gaussian_prior(x, beta, mu, sigma, thetamin, thetamax)
    return concat([theta, [beta]])


def non_repart_prior(hypercube):
    x = hypercube[:nDims]
    b = hypercube[-1]
    beta = betamin + (betamax-betamin)*b
    theta = power_gaussian_prior(
        x, beta, mu, sigmaLL, thetamin, thetamax)
    return concat([theta, [beta]])


def squeezed_repartitioned_prior(hypercube):
    x = hypercube[:ndims]
    b = hypercube[-1]
    beta = betamin + (5*betamax - betamin)*b
    theta = power_gaussian_prior(x, beta, mu, sigma, thetamin, thetamax)
    return concat([theta, [beta]])


# | Repartitioned loglikelihood
def repartitioned_loglikelihood(theta):
    t = theta[:nDims]
    beta = theta[-1]
    logl, phi = loglikelihood(t)
    logl += len(t) * -log(thetamax-thetamin)
    logl -= ln_rep_den(t, beta, mu, sigma, thetamin, thetamax).sum()
    return logl, phi


def non_repart_loglike(theta):
    t = theta[:nDims]
    beta = theta[-1]
    logl, phi = loglikelihood(t)
    logl += len(t) * -log(thetamax-thetamin)
    # logl -= ln_rep_den(t, beta, mu, sigma, thetamin, thetamax).sum()
    return logl, phi


def true_logZ():
    return nDims*(log(2*pi*sigma**2)/2.0 - log(thetamax-thetamin))


def bloated_gaussian_prior(hypercube):
    return GaussianPrior(mu, sigmaLL)(hypercube)


def power_gaussian_prior(hypercube, beta, mu, sigma, a, b):
    return mu + sqrt(2/beta)*sigma*erfinv((1-hypercube)*erf((a-mu)*sqrt(beta/2)/sigma) + hypercube*erf((b-mu)*sqrt(beta/2)/sigma))


def ln_rep_den(theta, beta, mu, sigma, a, b):
    return -beta * (theta-mu)**2/2/sigma**2 - log(pi*sigma**2/2/beta)/2 - log(erf((b-mu)*sqrt(beta/2)/sigma) - erf((a-mu)*sqrt(beta/2)/sigma))


def exec_polychord(series_name, settings, loglike, prior):
    print('running {}'.format(series_name))
    settings.file_root = series_name
    output = pypolychord.run_polychord(
        loglike, settings.nDims, settings.nDerived, settings, prior)
    sample = NestedSamples(root='./chains/{}'.format(settings.file_root))
    outputs[series_name] = output
    samples[series_name] = sample
    return output, sample


def overlay_histograms(samples_dict):
    for k in samples_dict.keys():
        if k in colors:
            plt.hist(samples_dict[k].logZ(1000), label=k, color=colors[k])
        else:
            plt.hist(samples_dict[k].logZ(1000), label=k)
    plt.legend()


def overlay_posteriors(samples_dict, data_range=(0, 1), transp=0.6):
    fig, axes = None, None
    for k in samples_dict.keys():
        if axes is None:
            if k in colors:
                fig, axes = samples_dict[k].plot_2d(
                    data_range, alpha=transp, color=colors[k])
            else:
                fig, axes = samples_dict[k].plot_2d(data_range, alpha=transp)
        else:
            if k in colors:
                samples_dict[k].plot_2d(axes, alpha=transp, color=colors[k])
            else:
                samples_dict[k].plot_2d(axes, alpha=transp, color=colors[k])


def main():
    exec_polychord('uniform', normal_settings, loglikelihood, uniform_prior)
    colors['uniform'] = 'green'
    exec_polychord('gaussian', normal_settings, loglikelihood, gaussian_prior)
    colors['gaussian'] = 'cyan'
    exec_polychord('bloated_gaussian', normal_settings,
                   loglikelihood, bloated_gaussian_prior)
    colors['bloated_gaussian'] = 'darkcyan'
    exec_polychord('ppr', repart_settings,
                   repartitioned_loglikelihood, repartitioned_prior)
    colors['ppr'] = 'red'
    # exec_polychord('squeezed_ppr', repart_settings,
    #                repartitioned_loglikelihood, repartitioned_prior)
    exec_polychord('ppnr', repart_settings,
                   non_repart_loglike, repartitioned_prior)
    colors['ppnr'] = 'orange'

    plt.subplots()
    overlay_histograms(samples)
    plt.show()


if __name__ == '__main__':
    main()
