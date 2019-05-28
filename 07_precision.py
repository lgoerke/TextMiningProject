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
from sklearn.metrics import classification_report

seed = 4715

i = datetime.now()
orig_stdout = sys.stdout
orig_stderr = sys.stderr
logname = 'log_prec_' + '001' +'.txt'
f = open(logname, 'w')
sys.stdout = f
sys.stderr = f

reviews = pickle.load(open('data/reviews.pkl','rb'))
prices = pickle.load(open('data/prices.pkl','rb'))

reviews_train = pickle.load(open('data/reviews_train.pkl','rb'))
reviews_test = pickle.load(open('data/reviews_test.pkl','rb'))
prices_train = pickle.load(open('data/prices_train.pkl','rb'))
prices_test = pickle.load(open('data/prices_test.pkl','rb'))

max_review_length = 500
reviews_train = sequence.pad_sequences(reviews_train, maxlen=max_review_length)
reviews_test = sequence.pad_sequences(reviews_test, maxlen=max_review_length)

print(len(reviews_test))

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(len(reviews_test), embedding_vecor_length, input_length=max_review_length))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# load the network weights
filename = "weights_l_003/weights-improvement-e02-l0.1267-a0.9537.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print(reviews_test.shape)

# Final evaluation of the model
#x = np.reshape(reviews_test[0], (1, len(reviews_test[0])))
predicted = model.predict(reviews_test, verbose=0)
#report = classification_report(prices_test, predicted)
#print(report)

sys.stderr = orig_stderr
sys.stdout = orig_stdout
f.close()