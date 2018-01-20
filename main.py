import pandas as pd
import re
import matplotlib.pyplot as plt
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
    print('data loaded')
    print("clean the text")
    row_it = data.iterrows()
    test_clean = []

    # Iterate the data
    for i, line in row_it:
        no_stopwords_tokens = []
        tokens = preprocess(line['text'])
        no_stopwords_tokens = clean_stopwords(tokens)
        test_clean.append(' '.join(no_stopwords_tokens))
    data['text_clean'] = test_clean

    # Explore gender distributation count
    print(data.gender.value_counts())

    # Explore distributation of words per gender
    Male = data[data['gender'] == 'male']
    Female = data[data['gender'] == 'female']
    Brand = data[data['gender'] == 'brand']
    Male_Words = pd.Series(' '.join(Male['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    Female_Words = pd.Series(' '.join(Female['text_clean'].astype(str)).lower().split(" ")).value_counts()[:20]
    Brand_words = pd.Series(' '.join(Brand['text_clean'].astype(str)).lower().split(" ")).value_counts()[:10]
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
    print(Brand_words)
    ts = Brand_words.plot(kind='bar', stacked=True, colormap='Paired')
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
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

#######************************  Neural network ******************************#######


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(x_train.data)))
model.add(Dense(len(x_train.data), activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(x_train.data, y_train.data, nb_epoch=100, batch_size=10)

# fix random seed for reproducibility
# numpy.random.seed(7)

# create the model
# embedding_vector_length = 32
# model = Sequential()
# model.add(Embedding(stop, embedding_vector_length, input_length=10))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128)

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
