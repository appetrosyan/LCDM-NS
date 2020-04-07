try:
    import tqdm

    def progressbar(items):
        return tqdm.tqdm(items)

except ImportError:
    def progressbar(items):
        return items
