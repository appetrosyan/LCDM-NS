import matplotlib.pyplot as plt
import numpy as np

quick_clean_run = {
    'noResume': True,
    'nLive': 20
}


def bench(data, test):
    for d in data:
        out = test(d)
    return []
