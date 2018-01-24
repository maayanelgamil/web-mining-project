import Q1
import Q2
import Q3
import Q4
# from __future__ import print_function
import pandas as pd
import os

os.environ["KERAS_BACKEND"] = "theano"
import numpy

#######************************   QUESTION 1 ******************************############################
clean_data = Q1.clean_q1('assets/gender-classifier.csv')

#######************************   QUESTION 2 ******************************############################
x_train, x_test, y_train, y_test = Q2.vectorizeAndGetTestAndTrain(clean_data)

# Q2.trainNeuralNetwork(x_train, x_test, y_train, y_test)
Q2.trainUsingNaiveBayes(x_train, x_test, y_train, y_test)
Q2.trainUsingKNN(x_train, x_test, y_train, y_test)

# Q2.TuneNaiveBayes(clean_data)
# Q2.TuneKNN(clean_data)

print("Finished tuning Naive bayes")
#######************************   QUESTION 3 ******************************############################

# Get data from twitter
# Q3.PrepareDataFromTwitter()

# Using the tweets csv file received by Q3.py
# For question 3 (Tweets)
q3_stop_words = ['#metoo', '#women', '#ladies', '#beard', '#men', '#bros', 'metoo', 'women', 'ladies', 'beard', 'men',
                 'bros']
clean_tweet_data = Q1.clean_q1('assets/tweetsNoReTweetsNoDuplicatedTweets.csv', added_stop_words=q3_stop_words)

#######************************   QUESTION 4 ******************************############################

x_test, y_test = Q2.vectorize(clean_tweet_data)
Q2.predictKNN(x_test, y_test)
Q2.predictNaiveBayes(x_test, y_test)
Q2.predictNeuralNetwork(x_test, y_test)
