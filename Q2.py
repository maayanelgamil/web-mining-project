import os
import string
from nltk.corpus import stopwords
os.environ["KERAS_BACKEND"] = "theano"
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'amp', 'get', 'gt', '1', '10', 'click']

# Declare the different models
knn = KNeighborsClassifier()
nb = MultinomialNB()
ann = Sequential()

# Declare the vectorizers
knnVectorizer = TfidfVectorizer()
nbVectorizer = TfidfVectorizer()
annVectorizer = TfidfVectorizer()


def getVectorizerForModel(model):
    vectorizer = None

    if "knn" in model:
        vectorizer = knnVectorizer
    elif "nb" in model:
        vectorizer = nbVectorizer
    elif "ann" in model:
        vectorizer = annVectorizer

    return vectorizer


def vectorize(data, model, mode):
    x = None
    vectorizer = getVectorizerForModel(model)

    if "train" in mode:
        x = vectorizer.fit_transform(data['text_clean'])
    elif "predict" in mode:
        x = vectorizer.transform(data['text_clean'])

    encoder = LabelEncoder()
    y = encoder.fit_transform(data['gender'])

    return x, y


def vectorizeAndSplitTestTrain(data, model, mode):
    x, y = vectorize(data, model, mode)
    # split into train and test sets
    return train_test_split(x, y, test_size=0.1)


def trainUsingNaiveBayes(x_train, x_test, y_train, y_test):
    nb.fit(x_train, y_train)
    pred = nb.predict(x_test)
    print("Navie Baies score:", nb.score(x_test, y_test))


def predictNaiveBayes(x_test, y_test):
    pred = nb.predict(x_test)
    print("Navie Baies score:", nb.score(x_test, y_test))

def predictNaiveBayesAfterTuning(train, test, tweet_stop_words):
    #Tune vectorizer according to the best parameters from the pipeline
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, max_df=0.3, stop_words=stop + tweet_stop_words)

    #Get test data and classification
    #encoder = LabelEncoder()
   # test_y = encoder.fit_transform(test['gender'])
    improved_features_train = vectorizer.fit_transform(train['text_clean'])
    improved_features_test = vectorizer.transform(test['text_clean'])

    clf = MultinomialNB(fit_prior=False, alpha=1)

    clf.fit(improved_features_train, train['gender'])
    pred = clf.predict(improved_features_test)

    score = metrics.accuracy_score(test['gender'], pred)
    print("improved accuracy:   %0.3f" % score)

### TUNE NAIVE BAYES #############
def TuneNaiveBayes(data):
    # split into train and test sets
    pipeline_x_train, pipeline_x_test, pipeline_y_train, pipeline_y_test = train_test_split(data['text_clean'],
                                                                                            data['gender'],
                                                                                            test_size=0.1)

    nb_clf = Pipeline([('vect', nbVectorizer), ('clf', MultinomialNB())])
    parameters = {'vect__max_df': (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 1.0),
                  'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                  'clf__alpha': (0.0001, 0.01, 1.0),
                  'clf__fit_prior': [True, False]}

    naive_clf = GridSearchCV(nb_clf, parameters)
    naive_clf = naive_clf.fit(pipeline_x_train, pipeline_y_train)

    print('Best score: ', naive_clf.best_score_)
    print('Best params: ', naive_clf.best_params_)


########### KNN #####################
def trainUsingKNN(x_train, x_test, y_train, y_test):
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)

    print("KNN score:", knn.score(x_test, y_test))


def predictKNN(x_test, y_test):
    knn.predict(x_test)

    print("KNN score:", knn.score(x_test, y_test))


def TuneKNN(data):
    ### TUNE KNN #####
    pipeline_x_train, pipeline_x_test, pipeline_y_train, pipeline_y_test = train_test_split(data['text_clean'],
                                                                                            data['gender'],
                                                                                            test_size=0.1)
    knn_clf = Pipeline([('vect', knnVectorizer), ('clf', KNeighborsClassifier())])

    knn_parameters = {'vect__max_df': (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 1.0),
                      'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                      'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                      'clf__leaf_size': (10, 20, 30, 40),
                      'clf__n_neighbors': (2, 5, 7),
                      'clf__weights': ['uniform', 'distance']}

    knn_gs = GridSearchCV(knn_clf, knn_parameters)
    knn_gs = knn_gs.fit(pipeline_x_train, pipeline_y_train)
    print('Best score: ', knn_gs.best_score_)
    print('Best params: ', knn_gs.best_params_)

def createModel():
    model_x = Sequential()
    print('Building model...')
    model_x.add(Dense(512, input_shape=(47215,)))
    model_x.add(Activation('relu'))
    model_x.add(Dropout(0.5))
    model_x.add(Dense(2))
    model_x.add(Activation('softmax'))
    model_x.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_x

def TuneNeuralNetwork(x_train, y_train):
    ### TUNE Neural Network #####
    from keras.wrappers.scikit_learn import KerasClassifier

    print("Start tuning NN")
    y_train = keras.utils.to_categorical(y_train, 2)
    print("create keras model")
    model = KerasClassifier(build_fn=createModel, verbose=0, epochs = 2)

    # define the grid search parameters
    batch_size = [5, 10, 50]
    epochs = [5, 10, 50]
    print("Creating grid")
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("Finished neural tuning")

#######************************  Neural Network ******************************#######

def trainNeuralNetwork(x_train, x_test, y_train, y_test):
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
    ann.add(Dense(512, input_shape=(47215,)))
    ann.add(Activation('relu'))
    ann.add(Dropout(0.5))
    ann.add(Dense(num_classes))
    ann.add(Activation('softmax'))

    ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = ann.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_split=0.1)
    score = ann.evaluate(x_test, y_test,
                         batch_size=batch_size, verbose=1)
    print(ann.metrics_names)
    print(score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def predictNeuralNetwork(x_test, y_test):
    num_classes = 2
    batch_size = 32
    y_test = keras.utils.to_categorical(y_test, num_classes)

    score = ann.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
    print('Neural Network score:', score[1])