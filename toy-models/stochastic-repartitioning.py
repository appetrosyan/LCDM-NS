#!/usr/bin/env python

from pypolychord.settings import PolyChordSettings
from pypolychord import run_polychord
from pypolychord.priors import UniformPrior, GaussianPrior
import numpy
from numpy import sqrt
from scipy.special import erfinv, erf
from anesthetic import NestedSamples
import fractions
import matplotlib.pyplot as plt
import time


class Model:
    def __init__(self, mu, cov, a, b, **kwargs):
        self.mu = numpy.array(mu)
        self.cov = numpy.array(cov)
        self.invCov = numpy.linalg.inv(cov)
        self.sigma = numpy.diag(self.cov)
        self.gaussian_norm = numpy.linalg.slogdet(2*numpy.pi*self.cov)[1]/2
        self.nDims = self.mu.size

        def unkwarg(arg, default):
            return default if arg not in kwargs else kwargs[arg]

        self.settings = PolyChordSettings(self.nDims, 0)
        self.settings.file_root = unkwarg('file_root', 'test')
        self.settings.nlive = unkwarg('live_points', 10**3)
        self.settings.read_resume = unkwarg('read_resume', False)
        self.settings.do_clustering = unkwarg('do_clustering', True)
        self.settings.feedback = unkwarg('feedback', 0)
        self.a = a
        self.b = b


mdl = Model([1, 2, 3], [[1, 0.6, 0.6],
                        [0.6, 1, 0.6],
                        [0.6, 0.6, 1]], a=-20, b=20, root_file='test', live_points=8**3)
cov1 = numpy.pad(mdl.cov, (0, 1))
cov1[-1, -1] = 1.0
mdl1 = Model([1, 2, 3, 0], cov1, a=-20, b=20, file_root='test-repart')


def true_logZ(modl):
    return modl.nDims*(numpy.log(2*numpy.pi*modl.sigma**2)/2.0 - numpy.log(modl.b-modl.a))


def logL(theta):
    delta = theta - mdl.mu
    ll = - mdl.gaussian_norm - numpy.linalg.multi_dot(
        [delta, mdl.invCov, delta])/2
    return ll, []


def simple_prior(point_in_hypercube):
    return UniformPrior(-20, 20)(point_in_hypercube)

# run_polychord(logL, mdl.nDims, 0, mdl.settings, simple_prior)


root2 = numpy.sqrt(2)


def PRprior(cube):
    x = cube[:-1]
    beta = cube[-1:].item()
    mu = mdl1.mu[:-1]
    sigma = mdl1.sigma[:-1]
    rr = sqrt(beta/2)
    prior = mu + root2 * \
        sigma * erfinv(
            x * erf(rr*(mdl1.b-mu) / sigma) +
            (1-x) * erf(rr*(mdl1.a-mu) / sigma)
        )/sqrt(beta)
    return numpy.append(prior, beta)


# run_polychord(logL, mdl1.nDims, 0, mdl1.settings, PRprior)

mdl.settings.file_root = 'testing-mixture'


def likelihood(params):
    theta = params[:-1]
    delta = theta - (mdl1.mu[:-1])
    # Gaussian norm is fine, by consturction the two things just multiply with one.
    ll = - mdl1.gaussian_norm - \
        numpy.linalg.multi_dot([delta, mdl1.invCov[:-1, :-1], delta])/2
    return ll, []


def mixture_model_prior(params):
    beta = params[-1:].item()
    rational_beta = fractions.Fraction(beta)
    theta = params[:-1]
    idx = hash(tuple(theta)) % rational_beta.denominator
    if idx > rational_beta.numerator:
        prior = UniformPrior(mdl1.a, mdl1.b)(theta)
    else:
        prior = GaussianPrior(mdl1.mu[:-1], mdl1.sigma[:-1])(theta)
    return numpy.append(prior, beta)


def gaussian_prior(params):
    beta = params[-1:].item()
    rational_beta = fractions.Fraction(beta)
    theta = params[:-1]
    idx = hash(tuple(theta)) % rational_beta.denominator
    prior = GaussianPrior(mdl1.mu[:-1], mdl1.sigma[:-1])(theta)
    return numpy.append(prior, beta)


def uniform_prior(params):
    beta = params[-1:].item()
    rational_beta = fractions.Fraction(beta)
    theta = params[:-1]
    idx = hash(tuple(theta)) % rational_beta.denominator
    prior = UniformPrior(mdl1.a, mdl1.b)(theta)
    return numpy.append(prior, beta)


def bloated_gaussian_prior(params):
    return numpy.append(GaussianPrior(mdl1.mu[:-1], 3*mdl1.sigma[:-1])(params[:-1]), params[-1:].item())


def offset_gaussian_prior(params):
    return numpy.append(GaussianPrior(mdl1.mu[:-1] + 0.3, mdl1.sigma[:-1])(params[:-1]), params[-1:].item())


def rev_offset_gaussian_prior(params):
    return numpy.append(GaussianPrior(mdl1.mu[:-1] - 0.3, mdl1.sigma[:-1])(params[:-1]), params[-1:].item())


def bench(func, *args, repeats=1):
    times = []
    for i in range(repeats):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end-start)
    return times


d = mdl1.nDims
options = mdl1.settings
options.nlive = 512


# from_gaussian = bench(run_polychord, likelihood, d, 0,
#                       options, gaussian_prior, repeats=20)
# For nlive=50
# [2.3464412689208984, 2.588460922241211, 3.0849390029907227, 3.9877099990844727, 1.9924259185791016, 1.9723868370056152, 3.391977071762085, 2.5541534423828125, 2.1456079483032227, 2.515599012374878,
#     1.9714350700378418, 2.3512470722198486, 3.4948441982269287, 2.002417802810669, 1.9801020622253418, 2.209686040878296, 2.020280122756958, 2.1705989837646484, 3.517896890640259, 1.9380548000335693]

# For nlive=512
# [18.582932949066162, 18.284988164901733, 18.52138376235962, 19.508468866348267, 18.576003074645996, 18.447359085083008, 18.272462844848633, 18.503297090530396, 18.506603002548218, 19.068424940109253, 18.338932037353516, 18.505760192871094, 18.639489889144897, 18.10427498817444, 18.52685809135437, 18.38833713531494, 18.256940126419067, 18.413666009902954, 18.548402070999146, 18.91140389442444]

# from_uniform = bench(run_polychord, likelihood, d, 0,
#                      options, uniform_prior, repeats=20)
# For nlive=50
# [6.031192064285278, 4.364820957183838, 6.278740167617798, 4.148439884185791, 6.5098302364349365, 4.950186729431152, 6.409754276275635, 4.955700874328613, 5.724398851394653, 5.772524118423462, 5.624842166900635, 5.077655076980591, 6.458392858505249, 4.727355241775513, 6.763023853302002, 3.500493049621582, 5.61159873008728, 5.94367790222168, 5.019721984863281, 5.0964741706848145]

# for nlive=512
# [30.120772123336792, 29.192168951034546, 34.522122859954834, 38.18403387069702, 44.97706317901611, 45.30061316490173, 37.02885675430298, 37.180575132369995, 40.587315797805786, 32.746140241622925, 30.404326915740967, 28.816236972808838, 31.317790985107422, 43.65755581855774, 41.22346806526184, 45.90019989013672, 29.843695163726807, 36.96680665016174, 33.012272119522095, 37.63551378250122]

# from_ppr = bench(run_polychord, likelihood, d, 0,
#                  options, PRprior, repeats=20)
# for nlive=50
# [2.5492191314697266, 2.678072214126587, 2.2829549312591553, 2.3218448162078857, 2.4961631298065186, 2.916872978210449, 2.3596138954162598, 2.6929938793182373, 2.2408578395843506, 2.8879880905151367, 2.408330202102661, 2.240095853805542, 2.662424087524414, 2.345767021179199, 2.3510117530822754, 2.2823920249938965, 2.6985690593719482, 2.3489580154418945, 2.3447391986846924, 2.4235379695892334]

# for nlive=512
# [22.410197973251343, 22.148893356323242, 22.428645133972168, 22.378524780273438, 22.478039979934692, 22.671180963516235, 22.617368936538696, 22.897464990615845, 22.708940029144287, 22.631688833236694, 22.425545930862427, 22.523781776428223, 22.688953161239624, 22.233675956726074, 22.2305850982666, 22.432966947555542, 22.33135986328125, 22.58033013343811, 22.07505202293396, 21.925750732421875]

# from_mixture = bench(run_polychord, likelihood, d, 0,
#                      options, mixture_model_prior, repeats=20)
# for nlive=50
# [2.2318079471588135, 2.2587890625, 2.2009570598602295, 2.0472140312194824, 2.491523027420044, 2.203134059906006, 2.812570095062256, 2.377692937850952, 2.267406702041626, 2.1818931102752686, 2.3163158893585205, 2.321192979812622, 2.432847738265991, 2.2643849849700928, 2.2828149795532227, 2.400684118270874, 2.709566831588745, 2.191816806793213, 2.3305797576904297, 2.2695229053497314]

# for nlive=512
# [21.301582098007202, 21.930859804153442, 22.009080171585083, 21.122539043426514, 21.079975128173828, 21.6564359664917, 22.570042848587036, 21.710810899734497, 21.9978129863739, 21.61481499671936, 21.323888301849365, 21.60486602783203, 21.93395495414734, 21.48937678337097, 22.27761721611023, 21.79326891899109, 22.073890924453735, 21.426614999771118, 22.318853855133057, 21.810243844985962]

options.file_root = 'gaussian'
run_polychord(likelihood, d, 0, options, gaussian_prior)

# run_polychord(likelihood, d, 0, options, offset_gaussian_prior)  # !!
# run_polychord(likelihood, d, 0, options, rev_offset_gaussian_prior)  # !!
# options.file_root = 'mixture'
# options.precision_criterion = 1e-2
# run_polychord(likelihood, d, 0, options, mixture_model_prior)
# options.file_root = 'PPR'
# run_polychord(likelihood, d, 0, options, PRprior)

# run_polychord(likelihood, d, 0, options, bloated_gaussian_prior)
# options.file_root = 'uniform'
# run_polychord(likelihood, d, 0, options, uniform_prior)
