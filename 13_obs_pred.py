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
import random

def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)

def lstm(print_number):

	seed = 4715

	i = datetime.now()
	orig_stdout = sys.stdout
	orig_stderr = sys.stderr
	logname = 'log_obs_' + print_number +'.txt'
	f = open(logname, 'w')
	sys.stdout = f
	sys.stderr = f

#	word_to_int = pickle.load(open('data/word_to_int.pkl','rb'))
#	int_to_word = pickle.load(open('data/int_to_word.pkl','rb'))

	char_to_int = pickle.load(open('data/char_to_int.pkl','rb'))
	int_to_char = pickle.load(open('data/int_to_char.pkl','rb'))
	
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
	all_prices= prices_train + prices_test
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
		seq_out =reviews_cheap[i + seq_length]
		X.append([seq_in])
		y.append(seq_out)
	
	dataX = X
	n_vocab = len(char_to_int)
	no_patterns = len(X)
	X = np.reshape(X, (no_patterns, seq_length, 1))
	X = X/n_vocab
	y = np_utils.to_categorical(y)

	model = Sequential()
	# withouth return sequences true for 002
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))

	# load the network weights
	filename = "weights_g_cheap_003/weights-improvement-e10-l2.9895.hdf5"
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())
	
	# pick a random seed
	#start = np.random.randint(0, len(dataX)-1)
	#pattern = dataX[start]
	#pattern = pattern[0]
	#print('Int_to_word: ',int_to_word) 
	pattern = np.arange(0,n_vocab,1)
	print(pattern)
	print('Seed: ')
	print('\"', ' '.join([int_to_char[value] for value in pattern]), '\"')
	# generate characters
	for i in range(n_vocab):
		yo = np.ones((seq_length,1))*pattern[i]
		#print(yo)
		print('--Input--')
		print(int_to_char[pattern[i]])
		x = np.reshape(yo, (1, seq_length, 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = np.argmax(prediction)
		p = random.random()
		ind = 0
		current_p = 0
		#while current_p < p:
		#	current_p = current_p + prediction[0,ind]
		#	ind = ind + 1
		#result = int_to_word[index]
		print('--PredVec--')
		pred = prediction.T
		print(pred.shape)
		pred = list(pred)
		sorted_pred = [i[0] for i in sorted(enumerate(pred), key=lambda x:x[1], reverse = True)]
		for j in range(10):
			print(int_to_char[sorted_pred[j]])
		#result = int_to_word[ind]
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		print('--- Prediction ---')
		#print(result, end=' ')
		print(result)
		#print('--- New Pattern ---')
		#pattern.append(index)
		#pattern.append(ind)
		#pattern = pattern[1:len(pattern)]
		#print('\"', ' '.join([int_to_char[value] for value in pattern]), '\"')
		print('--- New Iteration ---')
	print('\nDone.')

	sys.stderr = orig_stderr
	sys.stdout = orig_stdout
	f.close()

def printUsage():
	print("Usage: 04_generate.py {Number of run}")

def main(args):
	if not(len(args) == 1):
		printUsage()
	else:
		print_number = args[0]
		lstm(print_number)
 
if __name__ == "__main__":
	main(sys.argv[1:])

