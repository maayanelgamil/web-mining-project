from __future__ import print_function


import pandas as pd
import re
import os
import string
from nltk.corpus import stopwords
os.environ["KERAS_BACKEND"] = "theano"
import numpy

##################################################################################################################
#######************************   QUESTION 1 ******************************############################
##################################################################################################################


# Define regex consts
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)',  # anything else
]

# Create stop word dictionary
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'amp', 'get', 'gt', '1', '10', 'click']

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*', r'', s)
    return tokens_re.findall(s)


def preprocess(s):
    tokens = tokenize(s)
    # To lower
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def clean_stopwords(text):
    no_stopwords_tokens = []

    # Remove stop words
    for token in text:
        if token not in stop:
            no_stopwords_tokens.append(token)

    return no_stopwords_tokens


def clean_q1(corpus_path, added_stop_words=None):
    # Add stop words if needed
    if added_stop_words is not None:
        stop.extend(added_stop_words)

    global data
    data = pd.read_csv(corpus_path)

    # Drop unknown + brand gender + no gender
    data.dropna(subset=['gender'], how='all', inplace=True)
    data.drop(data[data.gender == 'brand'].index, inplace=True)
    data.drop(data[data.gender == 'unknown'].index, inplace=True)
    data.drop(data[pd.isnull(data.gender)].index, inplace=True)

    print('data loaded')
    print("clean the text")
    row_it = data.iterrows()
    test_clean = []

    # Iterate the data
    for i, line in row_it:
        no_stopwords_tokens = []
        tokens = preprocess(line['text'] + str(line['description']))
        no_stopwords_tokens = clean_stopwords(tokens)
        test_clean.append(' '.join(no_stopwords_tokens))
    data['text_clean'] = test_clean

    # Explore gender distributation count
    print(data.gender.value_counts())

    # Explore distributation of words per gender
    Male = data[data['gender'] == 'male']
    Female = data[data['gender'] == 'female']
    Male_Words = pd.Series(' '.join(Male['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    Female_Words = pd.Series(' '.join(Female['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    All_words = pd.Series(' '.join(data['text_clean'].astype(str)).lower().split(" ")).value_counts()[:10]

    print("**********FINISHED CLEANING THE TEXT***************")
    print(Female_Words)
    ts = Female_Words.plot(kind='bar', stacked=True, colormap='OrRd')
    ts.plot()
    # plt.show()
    print(Male_Words)
    ts = Male_Words.plot(kind='bar', stacked=True, colormap='plasma')
    ts.plot()
    # plt.show()
    print("**ALL WORDS**")
    print(All_words)
    ts = All_words.plot(kind='bar', stacked=True, colormap='Paired')
    ts.plot()
    # plt.show()

clean_q1('assets/gender-classifier.csv')

##################################################################################################################
#######************************   QUESTION 2 ******************************############################
##################################################################################################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Convolution1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# the Naive Bayes model
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['text_clean'])
encoder = LabelEncoder()
y = encoder.fit_transform(data['gender'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

nb = MultinomialNB()
nb.fit(x_train, y_train)
pred = nb.predict(x_test)
print("Navie Baies score:", nb.score(x_test, y_test))

### TUNE NAIVE BAYES #############
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# split into train and test sets
pipeline_x_train, pipeline_x_test, pipeline_y_train, pipeline_y_test = train_test_split(data['text_clean'], data['gender'], test_size=0.1)

nb_clf = Pipeline([('vect', vectorizer),('clf', MultinomialNB())])
parameters =  {'vect__max_df': (0.3,0.4,0.5,0.6,0.7, 0.75, 1.0),
               'vect__ngram_range': ((1, 1), (1, 2)),# unigrams or bigrams
               'clf__alpha': (0.0001, 0.01,1.0),
               'clf__fit_prior':[True, False] }

naive_clf = GridSearchCV(nb_clf, parameters)
naive_clf = naive_clf.fit(pipeline_x_train, pipeline_y_train)

print('Best score: ',naive_clf.best_score_)
print('Best params: ',naive_clf.best_params_)

########### KNN #####################
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print("KNN score:", knn.score(x_test, y_test))
### TUNE KNN #####
knn_clf = Pipeline([('vect', vectorizer),('clf', KNeighborsClassifier())])

knn_parameters = {'vect__max_df': (0.3,0.4,0.5,0.6,0.7, 0.75, 1.0),
               'vect__ngram_range': ((1, 1), (1, 2)),# unigrams or bigrams
               'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
               'clf__leaf_size': (10,20,30,40),
               'clf__n_neighbors': (2,5,7),
               'clf__weights': ['uniform', 'distance']}

knn_gs = GridSearchCV(knn_clf, knn_parameters)
knn_gs = knn_gs.fit(x_train.data, x_train.target)
print('Best score: ',knn_gs.best_score_)
print('Best params: ',knn_gs.best_params_)
#######************************  Neural Network ******************************#######

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 5

num_classes = 2
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix ''(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(29397,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print(model.metrics_names)
print(score)
print('Test score:', score[0])
print('Test accuracy:', score[1])
##################################################################################################################
#######************************   QUESTION 3 ******************************############################
##################################################################################################################

# Using the tweets csv file received by Q3.py

# For question 3 (Tweets)
# q3_stop_words = ['#metoo', '#women', '#ladies', '#beard', '#men', '#bros', 'metoo', 'women', 'ladies', 'beard', 'men', 'bros']
# clean_q1('assets/tweetsNoReTweets.csv', added_stop_words=q3_stop_words)

##################################################################################################################
#######************************   QUESTION 4 ******************************############################
##################################################################################################################
