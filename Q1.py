import string
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt


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
    # drop all low condifent rows - in twitter conffidence doesn't exists
    # This will improve our training to be more accurate
    if('gender:confidence' in data):
        data.drop(data[data['gender:confidence'] < 0.65].index, inplace=True)

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
    plt.show()
    print(Male_Words)
    ts = Male_Words.plot(kind='bar', stacked=True, colormap='plasma')
    ts.plot()
    plt.show()
    print("**ALL WORDS**")
    print(All_words)
    ts = All_words.plot(kind='bar', stacked=True, colormap='Paired')
    ts.plot()
    plt.show()
    return data