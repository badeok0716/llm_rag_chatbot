import pickle
def write_pkl(content, path):
    with open(path, "wb") as f:
        pickle.dump(content, f)