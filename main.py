import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import numpy

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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
    from nltk.corpus import stopwords
    import string
    # Create stop word dictionary
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']
    no_stopwords_tokens
    # Remove stop words
    for token in text:
        if token not in stop:
            no_stopwords_tokens.append(token)
    return no_stopwords_tokens;


def clean_q1(corpus_path):
    global data, no_stopwords_tokens
    data = pd.read_csv(corpus_path)
    print('data loaded')
    print("Part 1 - clean the text")
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
    plt.show()
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

print(nb.score(x_test, y_test))

#######************************   Nuiral network ******************************############################

os.environ["KERAS_BACKEND"] = "theano"

# fix random seed for reproducibility
numpy.random.seed(7)

# truncate and pad input sequences
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128)
