import numpy
# from matplotlib.pyplot import subplots, plot, show


def generate_clean_from_model(model, data_range):
    return data_range, model(data_range)


def generate_noisy_from_model(model, data_range, x_noise_amplitude=0.2, y_noise_amplitude=3):
    numpy.vectorize(model)
    r_data_range = data_range + numpy.random.random_sample(len(data_range)) * x_noise_amplitude
    r_model_range = model(data_range)
    r_model_range = r_model_range + numpy.random.random_sample(len(r_model_range)) * y_noise_amplitude
    return r_data_range, r_model_range


def mixture_model(model1, model2, data_range, **kwargs):
    _, mod1 = generate_noisy_from_model(model1, data_range[0::2], **kwargs)
    _, mod2 = generate_noisy_from_model(model2, data_range[1::2], **kwargs)
    r_model_range = numpy.empty((mod1.size + mod2.size), dtype=mod1.dtype)
    r_model_range[0::2] = mod1
    r_model_range[1::2] = mod2
    return data_range, r_model_range


# if __name__ == "__main__":
#     x = numpy.linspace(-10, 10, 50)
#     mdl1 = lambda x: 1.3*x**2
#     mdl2 = lambda x: x**2
#     x, y = mixture_model(mdl1, mdl2, x, x_noise_amplitude=0, y_noise_amplitude=0.3)
#     plot(x, y)
#     show()
