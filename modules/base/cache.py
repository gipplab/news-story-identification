from os.path import exists
from modules.preprocessing.io import load_pickle, write_pickle

def has_cache(cache_files):
    if isinstance(cache_files, list):
        for f in cache_files:
            if not exists(f):
                return False
        return True
    elif isinstance(cache_files, str):
        return exists(cache_files)
    else:
        raise Exception("cache_files must be string of list of strings")

def save_cache(objects, cache_files):
    if isinstance(cache_files, list) and isinstance(objects, list):
        if len(objects) != len(cache_files):
            raise Exception("Objects and files do not have same length")
        for i, f in enumerate(cache_files):
            print(f'Saving {f}...')
            write_pickle(obj=objects[i], path=f)
    elif isinstance(cache_files, str):
        print(f'Saving... {cache_files}...')
        write_pickle(obj=objects, path=cache_files)

def load_cache(cache_files):
    if isinstance(cache_files, list):
        objects = []
        for f in cache_files:
            print(f'Loading {f}...')
            objects.append(load_pickle(f))
        return objects
    elif isinstance(cache_files, str):
        print(f'Loading {cache_files}...')
        return load_pickle(cache_files)
    else:
        raise Exception("cache_files must be string of list of strings")
