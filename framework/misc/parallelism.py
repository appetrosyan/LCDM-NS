# from multiprocessing import Pool
try:
    from pathos.multiprocessing import ProcessingPool as Pool

    def parmap(f, x):
        with Pool() as p:
            rval = p.map(f, x)
        return rval

except ImportError:
    try:
        from multiprocessing.dummy import Pool as ThreadPool

        def parmap(f, x):
            with ThreadPool() as pool:
                results = pool.map(f, x)
            return results
    except ImportError:
        def parmap(f, x):
            return map(f, x)
