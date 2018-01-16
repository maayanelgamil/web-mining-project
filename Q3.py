import pandas as pd
import tweepy
from tweepy import OAuthHandler

# Find most popular country from train set
corpus_path = 'assets\gender-classifier-DFE-791531.csv'
data = pd.read_csv(corpus_path)

print(data['tweet_location'].value_counts())
print("As we can see, the most popular country is United States.")
print("We will chose this country for the twitter mining")

# Get all twits from the popular country

consumer_key = 'uW7b9X2txbXPgXXPjHNxk3yvZ'
consumer_secret = 'nxQ38FkZkkkf3OSrL1pZwYbRBXRhTeKVetQtpecxtOsW0R7Erz'
access_token = '33584667-xIa3A136SDuAk32JNqkZZYKqnwUFA2s30hKs8qUAp'
access_secret = '59nOtlV0fEuS9qeN0DeroMqsWQ45Qmcn0Os5IqmnNWDvd'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

places = api.geo_search(query="USA", granularity="country")
place_id = places[0].id

# tweets = api.search(q="place:%s" % place_id)

tweets = tweepy.Cursor(api.search,
                       q="place:%s" % place_id,
                       rpp=100,
                       result_type="recent",
                       include_entities=True,
                       lang="en").items(1000)

userId = []
text = []
place = []
description = []
df = pd.DataFrame()

for tweet in tweets:
    userId.append(tweet.user.id_str)
    text.append(tweet.text)
    place.append(tweet.place.name if tweet.place else "Undefined place")
    description.append(tweet.user.description if tweet.user.description else " ")

df["_unit_id"] = userId
df["place"] = place
df["text"] = text
df["description"] = description

df.to_csv("tweets.csv", encoding='utf-8')
