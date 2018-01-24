import pandas as pd
import tweepy
from tweepy import OAuthHandler

# Set the API keys & tokens for the tweepy API calls
consumer_key = 'uW7b9X2txbXPgXXPjHNxk3yvZ'
consumer_secret = 'nxQ38FkZkkkf3OSrL1pZwYbRBXRhTeKVetQtpecxtOsW0R7Erz'
access_token = '33584667-xIa3A136SDuAk32JNqkZZYKqnwUFA2s30hKs8qUAp'
access_secret = '59nOtlV0fEuS9qeN0DeroMqsWQ45Qmcn0Os5IqmnNWDvd'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

# Prepare a data frame with the following columns:
# userId  - the ID of the user that posted the tweet
# text - the tweet itself
# description - the description of that user's profile
userId = []
text = []
description = []
gender = []
df = pd.DataFrame()

# Set the tweepy API
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# Helper function to insert given tweets of a specific gender in the Data Frame
# Re-Tweets are excludes (usually bots & commercial)
def insertTweetsToLists(tweets, given_gender):
    tweetsCounter = 0

    for tweet in tweets:
        # stop if reached 500 tweets (we went for 500 male tweets and 500 for female)
        if tweetsCounter == 500:
            break

        # Exclude re-tweets (usually bots & commercial)
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            # Insert tweet to the Data Frame
            userId.append(tweet.user.id_str)
            text.append(tweet.text)
            description.append(tweet.user.description if tweet.user.description else " ")
            gender.append(given_gender)

        # Increase tweets counter by one
        tweetsCounter = tweetsCounter + 1


# Help function to get tweets by given queries for a specific gender.
# Then the function calls the function above to insert the tweets into lists
def getAndInsertTweetsByQuery(queries, given_gender):
    for query in queries:
        tweets = tweepy.Cursor(api.search, q=query, lang="en").items(500)
        insertTweetsToLists(tweets, given_gender)


def PrepareDataFromTwitter():
    # Search query for female tweets
    femaleQueries = ['#meToo', '#women', "#ladies"]

    # Use Cursor API call to find 500 tweets according to the given query
    getAndInsertTweetsByQuery(femaleQueries, 'female')

    # Search query for male tweets
    maleQueries = ['#beard', '#men', '#bros']

    # Use Cursor API call to find 500 tweets according to the given query
    getAndInsertTweetsByQuery(maleQueries, 'male')

    # Insert the data from tweets into the Data Frame
    # Naming the columns same as in Question 1 given train set
    df["_unit_id"] = userId
    df["text"] = text
    df["description"] = description
    df["gender"] = gender

    # Write the data frame to CSV file
    df.to_csv("tweetsNoReTweets.csv", encoding='utf-8')
