#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:28:35 2020

@author: haipro
"""
import unittest
import json
from keras.models import load_model
from nltk import ngrams
import numpy as np
import pandas as pd
import csv
import re
import string
import sys
from unidecode import unidecode

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#model = load_model("./model_train/spell_07_06_40.h5")

model = load_model("./model_train/spell_09_00_54 .h5")

#model = load_model("./model_train/spell_08_30_52_110820.h5")

model._make_predict_function()



MAXLEN=40
MAXNGRAM = 4
alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']
letters=list("abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÉÈẺẼẸÊẾỀỂỄỆÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴĐ")
accepted_char=list((string.digits + ''.join(letters)))

def extract_phrases(text):
    pattern = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(pattern, text)

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


def nltk_ngrams(words, n=5):
    return ngrams(words.split(), n)
      
def guess(ngram):
    text = ' '.join(ngram)
    preds = model.predict(np.array([encoder_data(text)]), verbose=0)
    return decoder_data(preds[0]).strip('\x00')

## ======================================================
# some common Vietnamese spell mistake
letters = list("abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÉÈẺẼẸÊẾỀỂỄỆÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴĐ")
letters2 = list("abcdefghijklmnopqrstuvwxyz")

typo={"ă":"aw","â":"aa","á":"as","à":"af","ả":"ar","ã":"ax","ạ":"aj","ắ":"aws","ổ":"oor","ỗ":"oox","ộ":"ooj","ơ":"ow",
"ằ":"awf","ẳ":"awr","ẵ":"awx","ặ":"awj","ó":"os","ò":"of","ỏ":"or","õ":"ox","ọ":"oj","ô":"oo","ố":"oos","ồ":"oof",
"ớ":"ows","ờ":"owf","ở":"owr","ỡ":"owx","ợ":"owj","é":"es","è":"ef","ẻ":"er","ẽ":"ex","ẹ":"ej","ê":"ee","ế":"ees","ề":"eef",
"ể":"eer","ễ":"eex","ệ":"eej","ú":"us","ù":"uf","ủ":"ur","ũ":"ux","ụ":"uj","ư":"uw","ứ":"uws","ừ":"uwf","ử":"uwr","ữ":"uwx",
"ự":"uwj","í":"is","ì":"if","ỉ":"ir","ị":"ij","ĩ":"ix","ý":"ys","ỳ":"yf","ỷ":"yr","ỵ":"yj","đ":"dd",
"Ă":"Aw","Â":"Aa","Á":"As","À":"Af","Ả":"Ar","Ã":"Ax","Ạ":"Aj","Ắ":"Aws","Ổ":"Oor","Ỗ":"Oox","Ộ":"Ooj","Ơ":"Ow",
"Ằ":"AWF","Ẳ":"Awr","Ẵ":"Awx","Ặ":"Awj","Ó":"Os","Ò":"Of","Ỏ":"Or","Õ":"Ox","Ọ":"Oj","Ô":"Oo","Ố":"Oos","Ồ":"Oof",
"Ớ":"Ows","Ờ":"Owf","Ở":"Owr","Ỡ":"Owx","Ợ":"Owj","É":"Es","È":"Ef","Ẻ":"Er","Ẽ":"Ex","Ẹ":"Ej","Ê":"Ee","Ế":"Ees","Ề":"Eef",
"Ể":"Eer","Ễ":"Eex","Ệ":"Eej","Ú":"Us","Ù":"Uf","Ủ":"Ur","Ũ":"Ux","Ụ":"Uj","Ư":"Uw","Ứ":"Uws","Ừ":"Uwf","Ử":"Uwr","Ữ":"Uwx",
"Ự":"Uwj","Í":"Is","Ì":"If","Ỉ":"Ir","Ị":"Ij","Ĩ":"Ix","Ý":"Ys","Ỳ":"Yf","Ỷ":"Yr","Ỵ":"Yj","Đ":"Dd"}

region={"ẻ":"ẽ","ẽ":"ẻ","ũ":"ủ","ủ":"ũ","ã":"ả","ả":"ã","ỏ":"õ","õ":"ỏ","i":"j", "à":"ồ"}
region2={"s":"x","l":"n","n":"l","x":"s","d":"gi","S":"X","L":"N","N":"L","X":"S","Gi":"D","D":"Gi"}

vowel=list("aeiouyáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵ")
acronym={"không":"ko"," anh":" a","em":"e","biết":"bít","giờ":"h","gì":"j","muốn":"mún","học":"hok","yêu":"iu",
         "chồng":"ck","vợ":"vk"," ông":" ô","được":"đc","tôi":"t",
         "Không":"Ko"," Anh":" A","Em":"E","Biết":"Bít","Giờ":"H","Gì":"J","Muốn":"Mún","Học":"Hok","Yêu":"Iu",
         "Chồng":"Ck","Vợ":"Vk"," Ông":" Ô","Được":"Đc","Tôi":"T",}

teen={"ch":"ck","ph":"f","th":"tk","nh":"nk",
      "Ch":"Ck","Ph":"F","Th":"Tk","Nh":"Nk", "Tr": "Ch"}

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
    

def add_noise(sentence, pivot1,pivot2):
    sentence=teen_code(sentence,0.5)
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

alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']


MAXLEN = 40


# ===============================================
    
def generate_data_only_noisy_non_encode(data, batch_size, pivot1, pivot2):
    x_non_decode = []
    for cur_index in range(0, len(data)):
      batch_size_noise = 0
      while batch_size_noise != batch_size:
            noise_text = add_noise(data[cur_index], pivot1, pivot2)
            if (noise_text not in x_non_decode):
                  x_non_decode.append(noise_text)
                  batch_size_noise += 1
    return x_non_decode


def warp_test_suite(testcase_class):
    """Load tests from a specific set of TestCase classes."""
    suite = unittest.TestSuite()
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(testcase_class)
    suite.addTest(tests)
    return suite

class TestDummy(unittest.TestCase):
    """A Dummy UnitTesting class."""

    def test_even(self):
#        """
#        Test that numbers between 0 and 5 are all even.
#        """
        self.assertEqual(1, 1)
        for i in range(1, len(data)):
            with self.subTest(i=i):
                preds = model.predict(np.array([encoder_data(data[i])]), verbose=0)
                self.assertEqual(decoder_data(preds[0]).strip('\x00'), CASE_TEST, 
                                 'Predict fail: {0}'.format(data[i]))


if __name__ == "__main__":

    # The test results dictionary, for the JSON.
    result_value = {"Failures": 0, "Errors": 0, "Skipped": 0, "Test Runs": 0, "Accuracy": 0}
    result_test = []
    with open("./Data/corpusViettel_3.txt") as f:
        reader = csv.reader(f)
        name_data = list(reader)
    for sub_name in name_data:
        BATCH_TEST_SIZE = 100
        CASE_TEST = sub_name[0]
        data = generate_data_only_noisy_non_encode([CASE_TEST], batch_size = BATCH_TEST_SIZE, pivot1 = 0.4, pivot2 = 0.986)
    
    #    data = [']
        # Setup and run the Test
        runner = unittest.TextTestRunner()
        TextTestResult = runner.run(warp_test_suite(TestDummy))

        # Passes the Result
        result_value["Failures"] += len(TextTestResult.failures)
        result_value["Errors"] += len(TextTestResult.errors)
        result_value["Skipped"] += len(TextTestResult.skipped)
        result_value["Test Runs"] += TextTestResult.testsRun
        result_value["Accuracy"] = (100 - result_value['Failures'])%100
        result_test.append([CASE_TEST, result_value["Accuracy"]])

    df = pd.DataFrame(result_test, columns=['CASE NAME', "accuracy"])    
    df.to_csv("./result.csv", sep="\t", encoding='utf-8', index=False)

#        
#if __name__ == "__main__":      
#    with open("./File_Test/thitran.csv") as f:
#        reader = csv.reader(f)
#        data = list(reader)
#    unittest.main() 