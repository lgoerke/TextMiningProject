### Import Keras ###
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import classification_report
from keras.utils.visualize_util import plot

### Import Utilities ###
from datetime import datetime
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os, os.path
import sys
import csv
import pickle

def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)

def lstm(print_number):

	seed = 4715

	i = datetime.now()
	orig_stdout = sys.stdout
	orig_stderr = sys.stderr
	logname = 'log_plot_gen_' + print_number +'.txt'
	f = open(logname, 'w')
	sys.stdout = f
	sys.stderr = f
	reviews = pickle.load(open('data/reviews.pkl','rb'))
	prices = pickle.load(open('data/prices.pkl','rb'))

	reviews_train = pickle.load(open('data/char_reviews_train.pkl','rb'))
	reviews_test = pickle.load(open('data/char_reviews_test.pkl','rb'))
	#reviews_train = pickle.load(open('data/reviews_train.pkl','rb'))
	#reviews_test = pickle.load(open('data/reviews_test.pkl','rb'))
	prices_train = pickle.load(open('data/prices_train.pkl','rb'))
	prices_test = pickle.load(open('data/prices_test.pkl','rb'))

	all_reviews = reviews_train + reviews_test
	#all_reviews = [item for sublist in all_reviews for item in sublist]
	all_prices = prices_train + prices_test
	#no_all = len(all_reviews)

	

	reviews_exp = []
	reviews_cheap = []
	for idx,rev in enumerate(all_reviews):
		if (all_prices[idx] == 1):
			reviews_exp.append(rev)
		else:
			reviews_cheap.append(rev)

	reviews_cheap = [item for sublist in reviews_cheap for item in sublist]
	no_all = len(reviews_cheap)

	seq_length = 100
	X = []
	y = []
	for i in range(0, no_all - seq_length, 1):
		seq_in = reviews_cheap[i:i + seq_length]
		seq_out = reviews_cheap[i + seq_length]
		X.append([seq_in])
		y.append(seq_out)

	# reshape X to be [samples, time steps, features]
	no_patterns = len(X)
	X = np.reshape(X, (no_patterns, seq_length, 1))
	X = X/no_patterns
	# one hot encode the output variable
	y = np_utils.to_categorical(y)
	max_review_length = 500

	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())	
	plot(model, to_file='model_gen_001.pdf', show_shapes=True, show_layer_names=False)

	sys.stderr = orig_stderr
	sys.stdout = orig_stdout
	f.close()

def printUsage():
	print("Usage: 02_lstm.py {Number of run}")

def main(args):
	if not(len(args) == 1):
		printUsage()
	else:
		print_number = args[0]
		lstm(print_number)
 
if __name__ == "__main__":
	main(sys.argv[1:])
