import pickle
import json
from os import listdir, remove
from os.path import isfile, join, exists
import pandas as pd

def unicode2ascii(text: str) -> str:
    return text.encode('ascii', 'ignore')

def list_files(folder, ext=None):
    files = [f for f in listdir(folder) if isfile(join(folder, f)) and (ext is None or (ext is not None and f.endswith(ext)))]
    return files

def read_csv_to_array(files):
    docs = []
    doc_ids = []
    for file in files:
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(file)
        for _, item in df.iterrows():
            docs.append(item['body'])
            doc_ids.append(item['id'])
    return docs, doc_ids

def read_csv_to_dataframe(files, usecols=['id', 'body']):
    df = pd.concat((pd.read_csv(f, usecols=usecols) for f in files))
    return df

def load_pickle(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    fp.close()
    return obj

def write_pickle(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()

def read_json(file):
    with open(file, 'r') as f:
        obj = json.load(f)
    f.close()
    return obj

def write_json(file, obj, indent=4):
    with open(file, 'w') as f:
        json.dump(obj, f, indent=indent)
    f.close()

def read_file(path, mode='r', ext='', encoding='utf-8'):
    if ext == 'json' or path.endswith('.json'):
        obj = load_json(path)
    elif ext == 'pkl' or path.endswith('.pkl'):
        obj = load_pickle(path)
    else:
        with open(path, mode=mode, encoding=encoding) as f:
            obj = f.read()
        f.close()
    return obj
    
def write_file(path, obj, mode='w', ext='', encoding='utf-8'):
    if ext == 'json' or path.endswith('.json'):
        write_json(path, obj)
    elif ext == 'pkl' or path.endswith('.pkl'):
        write_pickle(path, obj)
    else:
        with open(path, mode=mode, encoding=encoding) as f:
            f.write(obj)        
        f.close()

def update_file(path, obj, mode='w', ext='', encoding='utf-8'):
    if exists(path):
        remove(path)
    write_file(path, obj, mode=mode, ext=ext, encoding=encoding)
