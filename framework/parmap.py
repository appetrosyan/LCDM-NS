# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

def parmap(f, x):
    with Pool() as p:
        rval = p.map(f, x)
    return rval


