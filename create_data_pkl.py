from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import pickle

import os 
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))

dir_path = os.path.join(dir_path, 'Spell-correct/Data')
file_corpus = os.path.join(dir_path, 'corpusViettel.txt')

# =============================================================================
# def get_data(folder_path):
#     X = []
#     y = []
#     dirs = os.listdir(folder_path)
#     for path in tqdm(dirs):
#         file_paths = os.listdir(os.path.join(folder_path, path))
#         for file_path in tqdm(file_paths):
#             with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
#                 lines = f.readlines()
#                 lines = ' '.join(lines)
#                 X.append(lines)
#                 y.append(path)
# 
#     return X, y
# =============================================================================

def get_data_txt():
    with open(file_corpus, 'r', encoding="utf-8") as f:
        

get_data_txt()
train_path = os.path.join(dir_path, 'Train_Full')
X_data, y_data = get_data(train_path)


X_data, y_data = get_data_txt()


pickle.dump(X_data, open('/home/haipro/Documents/Project/Spell-correct/Data/VNTC_data.pkl', 'wb'))
    