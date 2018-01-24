import os
import string
from nltk.corpus import stopwords
os.environ["KERAS_BACKEND"] = "theano"
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import re
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy

vectorizer = TfidfVectorizer()
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'amp', 'get', 'gt', '1', '10', 'click']
def vectorizeAndGetTestAndTrain(data):
    # the Naive Bayes model
    x = vectorizer.fit_transform(data['text_clean'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['gender'])

    # split into train and test sets
    #x_train, x_test, y_train, y_test =
    return train_test_split(x, y, test_size=0.1)

def ClassifyUsingNaiveBayes(x_train, x_test, y_train, y_test):
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    pred = nb.predict(x_test)
    print("Navie Baies score:", nb.score(x_test, y_test))


### TUNE NAIVE BAYES #############
def TuneNaiveBayes(data):
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
def ClassifyUsingKNN(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)

    print("KNN score:", knn.score(x_test, y_test))


def TuneKNN(data):
    ### TUNE KNN #####
    pipeline_x_train, pipeline_x_test, pipeline_y_train, pipeline_y_test = train_test_split(data['text_clean'],
                                                                                            data['gender'],
                                                                                            test_size=0.1)
    knn_clf = Pipeline([('vect', vectorizer), ('clf', KNeighborsClassifier())])

    knn_parameters = {'vect__max_df': (0.3,0.4,0.5,0.6,0.7, 0.75, 1.0),
                   'vect__ngram_range': ((1, 1), (1, 2)),# unigrams or bigrams
                   'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                   'clf__leaf_size': (10,20,30,40),
                   'clf__n_neighbors': (2,5,7),
                   'clf__weights': ['uniform', 'distance']}

    knn_gs = GridSearchCV(knn_clf, knn_parameters)
    knn_gs = knn_gs.fit(pipeline_x_train, pipeline_y_train)
    print('Best score: ',knn_gs.best_score_)
    print('Best params: ',knn_gs.best_params_)

#######************************  Neural Network ******************************#######

def ClassifyWithNeuralNetwork():
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

def predcitWithBestResult(clean_data, test, added_stop_words = []):
    # the Naive Bayes model
    x = vectorizer.fit_transform(clean_data['text_clean'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(clean_data['gender'])

    # split into train and test sets
    x_train, y_train, x_test, y_test train_test_split(x, y, test_size=0.1)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.3, stop_words=stop + added_stop_words )
    clf = MultinomialNB(fit_prior=False, alpha=1)

    improved_features_train = vectorizer.fit_transform(x_train)

    encoder = LabelEncoder()
    test_data = data.drop(test['gender'].index, inplace=False)
    improved_features_test = vectorizer.transform(test_data)

    clf.fit(improved_features_train, y_train)
    pred = clf.predict(test_data)

    score = metrics.accuracy_score(test['gender'], pred)
    result_display.append(['Naive Bayes', 'Improved', score])