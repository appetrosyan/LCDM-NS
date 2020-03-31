import matplotlib.pyplot as plt
import numpy as np


def bench(data, tests):
    if not hasattr(tests, callable):
        raise ValueError('tests must be callable.')

    return []
