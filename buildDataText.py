import os
import settings
from sklearn.model_selection import train_test_split
import glob



import os 
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'Spell-correct/Data')
data_split_path = os.path.join(dir_path, 'VNTQcorpus-small.txt')
data_label = os.path.join(dir_path, 'Train_Full')

class FileReader(object):
    def __init__(self, filePath, encoder=None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s


class FileStore(object):
    def __init__(self, filePath=None, data=None):
        self.filePath = filePath
        self.data = data

    def StoreTextUnit(self, list):
        for index, value in enumerate(list):
            name_file = self.filePath + '/NH_(' + str(index) + ').txt'
            with open(name_file, 'w') as output:
                output.write(value)

if __name__ == '__main__':
    print('Reading Data Raw')
    json_train = FileReader(filePath=data_split_path).content()
    
    FileStore(filePath=data_label).StoreTextUnit(json_train)
