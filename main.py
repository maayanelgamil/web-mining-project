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

#######************************  Neural Network ******************************#######

from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
import json

train_x = data["text_clean"]
train_y = data["gender"]

train_x = train_x.as_matrix(columns=None)
train_y = train_y.as_matrix(columns=None)

for idx, val in enumerate(train_y):
    if val == 'male':
        train_y[idx] = 0
    else:
        train_y[idx] = 1

numpy.set_printoptions(threshold='nan')

# only work with the 3000 most popular words found in our dataset
max_words = 3000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

ignored = []


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    # return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            if word not in ignored:
                ignored.append(word)
    return wordIndices


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = numpy.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)


# Create the model

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

# Fit the model
model.fit(train_x, train_y,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

# Save the model

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

from keras.models import model_from_json

# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')

labels = ['male', 'female']

evalSentence = "Alon is the egg of all people"

# format your input for the neural net
testArr = convert_text_to_index_array(evalSentence)
input = tokenizer.sequences_to_matrix([testArr], mode='binary')
# predict which bucket your input belongs in
pred = model.predict(input)
# and print it for the humons
print("%s sentiment; %f%% confidence" % (labels[numpy.argmax(pred)], pred[0][numpy.argmax(pred)] * 100))

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
