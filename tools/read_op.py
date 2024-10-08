import pickle

def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def read_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def split_txt_by_scene(text):
    return 