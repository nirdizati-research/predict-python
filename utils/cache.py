import pickle


def load_from_cache(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_to_cache(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
