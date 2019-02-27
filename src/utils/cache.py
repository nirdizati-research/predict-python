import hashlib
import pickle


def get_digested(candidate_path):
    return hashlib.sha256(candidate_path.encode('utf-8')).hexdigest()


def load_from_cache(path, prefix=''):
    with open(prefix + get_digested(path) + '.pickle', 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_to_cache(path, obj, prefix=''):
    with open(prefix + get_digested(path) + '.pickle', "wb") as f:
        pickle.dump(obj, f)
