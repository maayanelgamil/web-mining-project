import Q1
import Q2
import Q3
# from __future__ import print_function
import pandas as pd
import os

os.environ["KERAS_BACKEND"] = "theano"
import numpy

#######************************   QUESTION 1 ******************************############################
clean_data = Q1.clean_q1('assets/gender-classifier.csv')

#######************************   QUESTION 2 ******************************############################

#ann_x_train, ann_x_test, ann_y_train, ann_y_test = Q2.vectorizeAndSplitTestTrain(clean_data, "ann", "train")
nb_x_train, nb_x_test, nb_y_train, nb_y_test = Q2.vectorizeAndSplitTestTrain(clean_data, "nb", "train")
#knn_x_train, knn_x_test, knn_y_train, knn_y_test = Q2.vectorizeAndSplitTestTrain(clean_data, "knn", "train")

#Q2.trainUsingKNN(knn_x_train, knn_x_test, knn_y_train, knn_y_test)
Q2.trainUsingNaiveBayes(nb_x_train, nb_x_test, nb_y_train, nb_y_test)
#Q2.trainNeuralNetwork(ann_x_train, ann_x_test, ann_y_train, ann_y_test)


# Q2.TuneNaiveBayes(clean_data)
# Q2.TuneKNN(clean_data)
#Q2.TuneNeuralNetwork(ann_x_train, ann_y_train)

print("Finished tuning Naive bayes")
#######************************   QUESTION 3 ******************************############################

# Get data from twitter
# Q3.PrepareDataFromTwitter()

# Using the tweets csv file received by Q3.py
# For question 3 (Tweets)
q3_stop_words = ['#metoo', '#women', '#ladies', '#beard', '#men', '#bros', 'metoo', 'women', 'ladies', 'beard', 'men','bros']
clean_tweet_data = Q1.clean_q1('assets/tweetsNoReTweetsNoDuplicatedTweets.csv', added_stop_words=q3_stop_words)

#######************************   QUESTION 4 ******************************############################

#nb_x_test, nb_y_test = Q2.vectorize(clean_tweet_data, "nb", "predict")

Q2.predictNaiveBayesAfterTuning(clean_data,clean_tweet_data , q3_stop_words)
