import Q1
import Q2
import Q3
import Q4
#from __future__ import print_function
import pandas as pd
import os

os.environ["KERAS_BACKEND"] = "theano"
import numpy

#######************************   QUESTION 1 ******************************############################
clean_data = Q1.clean_q1('assets/gender-classifier.csv')

#######************************   QUESTION 2 ******************************############################
x_train, x_test, y_train, y_test = Q2.vectorizeAndGetTestAndTrain(clean_data)
Q2.ClassifyUsingNaiveBayes(x_train, x_test, y_train, y_test)
Q2.ClassifyUsingKNN(x_train, x_test, y_train, y_test)

Q2.TuneNaiveBayes(clean_data)
#######************************   QUESTION 3 ******************************############################

# Using the tweets csv file received by Q3.py
#For question 3 (Tweets)
q3_stop_words = ['#metoo', '#women', '#ladies', '#beard', '#men', '#bros', 'metoo', 'women', 'ladies', 'beard', 'men', 'bros']
Q1.clean_q1('assets/tweetsNoReTweets.csv', added_stop_words=q3_stop_words)

#######************************   QUESTION 4 ******************************############################

