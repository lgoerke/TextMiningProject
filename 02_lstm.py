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
	logname = 'log_l_' + print_number +'.txt'
	f = open(logname, 'w')
	sys.stdout = f
	sys.stderr = f

	reviews = pickle.load(open('data/reviews.pkl','rb'))
	prices = pickle.load(open('data/prices.pkl','rb'))

	reviews_train = pickle.load(open('data/reviews_train.pkl','rb'))
	reviews_test = pickle.load(open('data/reviews_test.pkl','rb'))
	prices_train = pickle.load(open('data/prices_train.pkl','rb'))
	prices_test = pickle.load(open('data/prices_test.pkl','rb'))
	
	int_to_word = pickle.load(open('data/int_to_word.pkl','rb'))
	int_to_char = pickle.load(open('data/int_to_char.pkl','rb'))


	print('Number of reviews: ',len(reviews))
	print('Number of vocab: ', len(int_to_word))
	

	max_review_length = 500
	reviews_train = sequence.pad_sequences(reviews_train, maxlen=max_review_length)
	reviews_test = sequence.pad_sequences(reviews_test, maxlen=max_review_length)

	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(len(reviews), embedding_vecor_length, input_length=max_review_length))
	model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(pool_length=2))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	# define the checkpoint
	filepath= 'weights_l_' + str(print_number) + '/weights-improvement-e{epoch:02d}-l{loss:.4f}-a{acc:.4f}.hdf5'
	mkdir_p(os.path.dirname(filepath))
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
	callbacks_list = [checkpoint]

	model.fit(reviews_train, prices_train, nb_epoch=3, batch_size=64, callbacks=callbacks_list)
	# Final evaluation of the model
	predicted = model.predict(reviews_test, verbose=0)
	scores = model.evaluate(reviews_test, prices_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	pathpred= 'data/predicted_classes_l_'+ str(print_number) + '.pkl'
	pickle.dump(predicted,open(pathpred,'wb'))
	print(predicted)
	#report = classification_report(prices_test, predicted)
	#print(report)

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



