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
	logname = 'log_plot_' + print_number +'.txt'
	f = open(logname, 'w')
	sys.stdout = f
	sys.stderr = f

	reviews = pickle.load(open('data/reviews.pkl','rb'))

	max_review_length = 500

	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(len(reviews), embedding_vecor_length, input_length=max_review_length))
	#model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
	#model.add(MaxPooling1D(pool_length=2))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	plot(model, to_file='model_002.pdf', show_shapes=True, show_layer_names=False)

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



