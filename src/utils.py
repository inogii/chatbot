import glob
import os
import pickle

from IPython.display import Image, display
from sentence_transformers import SentenceTransformer


def pickler(filename, data):
    with open(f'{filename}', 'wb') as file:
        pickle.dump(data, file)
        print(f'Data saved to {filename}')

def unpickler(filename):
    # check if file exists
    if os.path.exists(filename):
        with open(f'{filename}', 'rb') as file:
            data = pickle.load(file)
            return data
    return None



def add_keywords(img_ocr):
    return f'\nImage Keywords: {img_ocr}'

def is_cached(model_id):
    model_id = model_id.replace('/', '_')
    models = os.listdir('models')
    return model_id in models

def load(model_id):
    os.makedirs('models', exist_ok=True)
    if not is_cached(model_id):
        print('Downloading...')
    
    model = SentenceTransformer(model_id, cache_folder='models')
    print(f'Model {model_id} loaded.')
    return model

def find_imgs(page:int, data_dir:str):
    imgs = [ file for file in glob.glob(str(os.path.join({data_dir}, f"{page}_*"))) ]
    return imgs

def show_imgs(page:int, data_dir:str):
    imgs = find_imgs(page, data_dir=data_dir)
    for img in imgs:
        display(Image(filename=img))


