# from multiprocessing import Pool
try:
    from pathos.multiprocessing import ProcessPool as Pool


    def parmap(f, x):
        with Pool() as p:
            result = p.map(f, x)
        return result

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
