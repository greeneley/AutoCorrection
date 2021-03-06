# -*- coding: utf-8 -*-
"""Train_Spell_Checker_2.0

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LMsDiGGSF7LvMTB2tz4XOsal2Az5pVGz
"""

!pip install tensorflow==1.15
!pip install tensorflow-gpu==1.15

!pip install keras==2.3.1
!pip install unidecode

import re
from collections import Counter
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk import ngrams, word_tokenize
import numpy as np
import re
from unidecode import unidecode
import string
from tqdm import tqdm
from random import shuffle
from datetime import datetime
import os
import time
import re
import random
from sklearn.model_selection import train_test_split
# Build the neural network
# this is adapted from the seq2seq architecture, which can be used for Machine Translation, Text Summarization Image Captioning ...
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense,LSTM, Bidirectional
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# BASIC INSTALL 
# KEY: 4/2gFyJvqhPILwIRMw0tT1EIh97zJx4LqRiUE8rhhQ3kDp8_Auvq-AlWA
!apt-get update -qq 2>&1 > /dev/null
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse 
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive/Colab
# %cd Spell-Checker-master/

!ls

# some common Vietnamese spell mistake
letters = list("abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÉÈẺẼẸÊẾỀỂỄỆÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴĐ")
letters2 = list("abcdefghijklmnopqrstuvwxyz")
alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']
typo = {"ă":"aw","â":"aa","á":"as","à":"af","ả":"ar","ã":"ax","ạ":"aj","ắ":"aws","ổ":"oor","ỗ":"oox","ộ":"ooj","ơ":"ow",
"ằ":"awf","ẳ":"awr","ẵ":"awx","ặ":"awj","ó":"os","ò":"of","ỏ":"or","õ":"ox","ọ":"oj","ô":"oo","ố":"oos","ồ":"oof",
"ớ":"ows","ờ":"owf","ở":"owr","ỡ":"owx","ợ":"owj","é":"es","è":"ef","ẻ":"er","ẽ":"ex","ẹ":"ej","ê":"ee","ế":"ees","ề":"eef",
"ể":"eer","ễ":"eex","ệ":"eej","ú":"us","ù":"uf","ủ":"ur","ũ":"ux","ụ":"uj","ư":"uw","ứ":"uws","ừ":"uwf","ử":"uwr","ữ":"uwx",
"ự":"uwj","í":"is","ì":"if","ỉ":"ir","ị":"ij","ĩ":"ix","ý":"ys","ỳ":"yf","ỷ":"yr","ỵ":"yj","đ":"dd",
"Ă":"Aw","Â":"Aa","Á":"As","À":"Af","Ả":"Ar","Ã":"Ax","Ạ":"Aj","Ắ":"Aws","Ổ":"Oor","Ỗ":"Oox","Ộ":"Ooj","Ơ":"Ow",
"Ằ":"AWF","Ẳ":"Awr","Ẵ":"Awx","Ặ":"Awj","Ó":"Os","Ò":"Of","Ỏ":"Or","Õ":"Ox","Ọ":"Oj","Ô":"Oo","Ố":"Oos","Ồ":"Oof",
"Ớ":"Ows","Ờ":"Owf","Ở":"Owr","Ỡ":"Owx","Ợ":"Owj","É":"Es","È":"Ef","Ẻ":"Er","Ẽ":"Ex","Ẹ":"Ej","Ê":"Ee","Ế":"Ees","Ề":"Eef",
"Ể":"Eer","Ễ":"Eex","Ệ":"Eej","Ú":"Us","Ù":"Uf","Ủ":"Ur","Ũ":"Ux","Ụ":"Uj","Ư":"Uw","Ứ":"Uws","Ừ":"Uwf","Ử":"Uwr","Ữ":"Uwx",
"Ự":"Uwj","Í":"Is","Ì":"If","Ỉ":"Ir","Ị":"Ij","Ĩ":"Ix","Ý":"Ys","Ỳ":"Yf","Ỷ":"Yr","Ỵ":"Yj","Đ":"Dd"}

region = {"ẻ":"ẽ","ẽ":"ẻ","ũ":"ủ","ủ":"ũ","ã":"ả","ả":"ã","ỏ":"õ","õ":"ỏ","i":"j"}

region2 = {"s":"x","l":"n","n":"l","x":"s","d":"gi","S":"X","L":"N","N":"L","X":"S","Gi":"D","D":"Gi"}

vowel = list("aeiouyáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵ")

acronym = {"không":"ko"," anh":" a","em":"e","biết":"bít","giờ":"h","gì":"j","muốn":"mún","học":"hok","yêu":"iu",
         "chồng":"ck","vợ":"vk"," ông":" ô","được":"đc","tôi":"t",
         "Không":"Ko"," Anh":" A","Em":"E","Biết":"Bít","Giờ":"H","Gì":"J","Muốn":"Mún","Học":"Hok","Yêu":"Iu",
         "Chồng":"Ck","Vợ":"Vk"," Ông":" Ô","Được":"Đc","Tôi":"T",}

teen = {"ch":"ck","ph":"f","th":"tk","nh":"nk",
      "Ch":"Ck","Ph":"F","Th":"Tk","Nh":"Nk", "Tr": "Ch","Tr":"Ch", "tr":"ch"}
# function for adding mistake( noise)
def teen_code(sentence,pivot):
    random = np.random.uniform(0,1,1)[0]
    new_sentence=str(sentence)
    if random>pivot:
        for word in acronym.keys():
            if re.search(word, new_sentence):
                random2 = np.random.uniform(0,1,1)[0]
                if random2 <0.5:
                    new_sentence=new_sentence.replace(word,acronym[word])
        for word in teen.keys(): 
            if re.search(word, new_sentence):
                random3 = np.random.uniform(0,1,1)[0]
                if random3 <0.05:
                    new_sentence=new_sentence.replace(word,teen[word])        
        return new_sentence
    else:
        return sentence
    
def add_noise(sentence, pivot1, pivot2):
    sentence = teen_code(sentence,0.4)
    noisy_sentence = ""
    i = 0
    while i < len(sentence):
        if sentence[i] not in letters:
            noisy_sentence+=sentence[i]
        else: 
            random = np.random.uniform(0,1,1)[0]   
            if random < pivot1:
                noisy_sentence+=(sentence[i])
            elif random<pivot2:
                if sentence[i] in typo.keys() and sentence[i] in region.keys():
                    random2=np.random.uniform(0,1,1)[0]
                    if random2<=0.4:
                        noisy_sentence+=typo[sentence[i]]
                    elif random2<0.8:
                        noisy_sentence+=region[sentence[i]]
                    elif random2<0.95 :
                        noisy_sentence+=unidecode(sentence[i])
                    else:
                        noisy_sentence+=sentence[i]
                elif sentence[i] in typo.keys():
                    random3=np.random.uniform(0,1,1)[0]
                    if random3<=0.6:
                        noisy_sentence+=typo[sentence[i]]
                    elif random3<0.9 :
                        noisy_sentence+=unidecode(sentence[i])                        
                    else:
                        noisy_sentence+=sentence[i]
                elif sentence[i] in region.keys():
                    random4=np.random.uniform(0,1,1)[0]
                    if random4<=0.6:
                        noisy_sentence+=region[sentence[i]]
                    elif random4<0.85 :
                        noisy_sentence+=unidecode(sentence[i])                        
                    else:
                        noisy_sentence+=sentence[i]
                elif i<len(sentence)-1 :
                    if sentence[i] in region2.keys() and (i==0 or sentence[i-1] not in letters) and sentence[i+1] in vowel:
                        random5=np.random.uniform(0,1,1)[0]
                        if random5<=0.9:
                            noisy_sentence+=region2[sentence[i]]
                        else:
                            noisy_sentence+=sentence[i]
                    else:
                        noisy_sentence+=sentence[i]

            else:
                new_random = np.random.uniform(0,1,1)[0]
                if new_random <=0.33:
                    if i == (len(sentence) - 1):
                        continue
                    else:
                        noisy_sentence+=(sentence[i+1])
                        noisy_sentence+=(sentence[i])
                        i += 1
                elif new_random <= 0.66:
                    random_letter = np.random.choice(letters2, 1)[0]
                    noisy_sentence+=random_letter
                else:
                    pass
      
        i += 1
    return noisy_sentence

# So a 5-grams contain at most 7*5 = 35 character (except one that has spell mistake)
# add "\x00" padding at the end of 5-grams in order to equal their length

MAXLEN = 40

def encoder_data(text, maxlen=MAXLEN):
    text = "\x00" + text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
      for j in range(i+1, maxlen):
        x[j, 0] = 1
    return x
  
def decoder_data(x):
    x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)



def generate_data(data, batch_size, pivot1, pivot2 ):
  while True:
    for cur_index in range(len(data)):
        x, y = [], []
        for i in range(batch_size):  
            y.append(encoder_data(data[cur_index]))
            x.append(encoder_data(add_noise(data[cur_index], pivot1, pivot2)))
        yield np.array(x), np.array(y)

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

text_file = open("./Data/corpusViettel_3.txt", "r")
lines = text_file.read().split('\n')
lines = [line.lower().strip() for line in lines]
# lines = [lines[i:i + 20] for i in range(0, len(lines), 20)]
lines.remove('')

def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)

import itertools
phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

print(len(phrases))
print(phrases[-10:])

phrases

NGRAM = 5 
MAXLEN = 40
def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)

# list_ngrams = []
# NGRAM = 5 
# MAXLEN = 40

# for p in tqdm(lines):
#   for _ in range(2, MAXLEN+1):
#     for ngr in gen_ngrams(p, _):
#       print(ngr)
#       # if len(" ".join(ngr)) < MAXLEN:
#       #   list_ngrams.append(" ".join(ngr))

# extract Latin- characters only
alphabet_special = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'
MAXNGRAM = 5 
MAXLEN = 40
def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)
   
list_ngrams = []
for p in tqdm(phrases):
  if not re.match(alphabet_special, p.lower()):
    continue
  for NGRAM in range(2, MAXNGRAM + 1):
    for ngr in gen_ngrams(p, NGRAM):
      if len(" ".join(ngr)) < MAXLEN:
        list_ngrams.append(" ".join(ngr))

len(list_ngrams)

list_ngrams = list(set(list_ngrams))

list_ngrams

BATCH_SIZE = 512
BATCH_VALID_SIZE = 256
now = datetime.now()
current_time = now.strftime("%H:%M:%S_%d%m%y")
# train_data = ["Đà Nẵng", "Sơn Trà", "bệnh viện", "Hoàng Diệu", "Nguyễn Văn Linh", "Nhà Hàng", "Nguyễn Thái Học", "Khách sạn", "Ngũ Hành Sơn"]
# train_data = ["Đà Nẵng", "Sơn Trà"]
# valid_data = ["Đà Nẵng", "Sơn Trà"]
train_data = lines
start_time = time.time()

train_generator = generate_data(train_data, batch_size = BATCH_SIZE, pivot1 = 0.4, pivot2 = 0.986)
validation_generator = generate_data(train_data, batch_size = BATCH_SIZE, pivot1 = 0.84, pivot2 = 0.986)

print("--- %s seconds ---" % (time.time() - start_time))

import pandas as pd
df = pd.DataFrame(data_non_encode)    
df.to_csv("./generator.csv", sep="\t", encoding='utf-8', index=False)

train_data

train_non_encode

"""model spell.h5 + ["Đà Nẵng", " Srứaơn Trà"]: spell_09:04:45_060820.h5

model spell.h5 + ["Đà Nẵng", "Sơn Trà", "Bệnh Viện"]: spell_07:55:18_070820.h5

model spell.h5 + 19 loại dữ liệu: spell_09:16:05_070820.h5


### 11/08/2020
["Đà Nẵng", "Sơn Trà", "bệnh viện", "Hoàng Diệu", "Nguyễn Văn Linh", "Nhà Hàng", "Nguyễn Thái Học", "Khách sạn", "Ngũ Hành Sơn"] : spell_02:50:48_120820.h5

Model new (rất tốt): spell_07_49_48_120820.h5
"""

encoder = LSTM(256, input_shape=(MAXLEN, len(alphabet)), return_sequences=True) # input_shape: 40x199
decoder = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))
model = Sequential()
model.add(encoder)
model.add(decoder)
model.add(TimeDistributed(Dense(256)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

from keras import backend as K

filepath = "./model/spell.h5"
model = load_model(filepath)

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S_%d%m%y")
start_time = time.time()
checkpointer = ModelCheckpoint(filepath=os.path.join('./model/spell_{}.h5'.format(current_time)), save_best_only=True, verbose=1)
seqModel = model.fit_generator(train_generator, steps_per_epoch = len(train_data), epochs=10,
                    validation_data = validation_generator, validation_steps = len(train_data), callbacks = [checkpointer])
plot_model_history(seqModel)  
print("--- %s seconds ---" % (time.time() - start_time))

preds = model.predict(np.array([encoder_data("binh viện")]), verbose=0)
print(decoder_data(preds[0]).strip('\x00'))