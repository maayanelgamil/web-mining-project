import pandas as pd;
import re
import matplotlib

corpus_path = 'D:\gender-classifier-DFE-791531.csv'
data = pd.read_csv(corpus_path)

print('data loaded')
print("Part 1 - clean the text")

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
    # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
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

print(Female_Words)
Female_Words.plot(kind='bar', stacked=True, colormap='OrRd')
print(Male_Words)
Male_Words.plot(kind='bar', stacked=True, colormap='plasma')
print(Brand_words)
Brand_words.plot(kind='bar', stacked=True, colormap='Paired')

print("**********FINISHED CLEANING THE TEXT***************")
