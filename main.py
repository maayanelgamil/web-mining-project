import sklearn.datasets as sk
import pandas as pd;
from time import time
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import preprocessing
import re

corpus_path = 'D:\gender-classifier-DFE-791531.csv'
data = pd.read_csv(corpus_path)

print('data loaded')

print("Part 1 - clean the text")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

data['Tweets'] = [cleaning(s) for s in data['text']]
data['Description'] = [cleaning(s) for s in data['description']]

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
data['Tweets'] = data['Tweets'].str.lower().str.split()
data['Tweets'] = data['Tweets'].apply(lambda x : [item for item in x if item not in stop])

data.gender.value_counts()

Male = data[data['gender'] == 'male']
Female = data[data['gender'] == 'female']
Brand = data[data['gender'] == 'brand']
Male_Words = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]
Female_Words = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:20]
Brand_words = pd.Series(' '.join(Brand['Tweets'].astype(str)).lower().split(" ")).value_counts()[:10]


print(Female_Words)

print(Male_Words)
    # for fileB, target in zip(data.data, data.target):
    #     # ------------TO LOWER CASE-------------
    #     file = fileB.decode("utf-8").lower()
    #     # ------------TOKENIZE-------------
    #     word_tokens = word_tokenize(file)
    #     # ------------REMOVE STOP WORDS-------------
    #     filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #     filtered_sentence = []
    #     for w in word_tokens:
    #         if w not in stop_words:
    #             filtered_sentence.append(w)
    #     # ------------STEMMING-------------
    #     ps = PorterStemmer()
    #     stemmedFile = []
    #     for word in filtered_sentence:
    #         for w in word.split(" "):
    #             stem = ps.stem(w)
    #             stemmedFile.append(stem)
    #             term_per_category[target][word] += 1
    #     # ------------PUT FILE BACK-------------
    #     fileB = ' '.join(stemmedFile)
    #

print("**********FINISHED CLEANING THE TEXT***************")
