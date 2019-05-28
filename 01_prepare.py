#-*- coding: utf-8 -*-

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

def make_binary(price_list,review_list):
	bin_prices = []
	bin_reviews = []
	for i,entry in enumerate(price_list):
		if entry < 15:
			bin_prices.append(0)
			bin_reviews.append(review_list[i])
		elif entry > 50:
			bin_prices.append(1)
			bin_reviews.append(review_list[i])
		else:
			pass
	return bin_prices, bin_reviews

def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)

def strip_punct(all_reviews):
	all_reviews = all_reviews.replace('.','')
	all_reviews = all_reviews.replace(',','')
	all_reviews = all_reviews.replace(':','')
	all_reviews = all_reviews.replace(';','')
	all_reviews = all_reviews.replace('"','')
	all_reviews = all_reviews.replace(')','')
	all_reviews = all_reviews.replace('(','')
	all_reviews = all_reviews.replace('#','')
	all_reviews = all_reviews.replace('?','')
	all_reviews = all_reviews.replace('!','')
	all_reviews = all_reviews.replace('&',' ')
	all_reviews = all_reviews.replace('/',' ')
	all_reviews = all_reviews.replace('+',' ')
	all_reviews = all_reviews.replace('‚','')
	all_reviews = all_reviews.replace('”','')
	all_reviews = all_reviews.replace('“','')
	return all_reviews

seed = 4715

reviews = []
prices = []

# Read reviews and prices from csv file
with open('metamatrix3b.random.csv') as csvfile:
	reviewreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	for row in reviewreader:
		if not (row[7]=='unk'):
			prices.append(row[7])
			reviews.append(row[20].lower())

print('Number of reviews before',len(reviews))
prices = list(map(float, prices))
prices, reviews = make_binary(prices, reviews)
print('Number of reviews is ' ,len(prices), ' = ', len(reviews) )
#print(reviews[0])
#print(prices[10])
#print(reviews[10])

all_reviews = ' '.join(reviews)

chars = sorted(list(set(all_reviews)))
n_vocab = len(chars)

all_rev = all_reviews.split(' ')
all_words = set(all_rev)
n_words = len(all_words)
print('Total Words With Punctuation: ', n_words)

all_reviews = all_reviews.replace('.','')
all_reviews = all_reviews.replace(',','')
all_reviews = all_reviews.replace(':','')
all_reviews = all_reviews.replace(';','')
all_reviews = all_reviews.replace('"','')
all_reviews = all_reviews.replace(')','')
all_reviews = all_reviews.replace('(','')
all_reviews = all_reviews.replace('#','')
all_reviews = all_reviews.replace('?','')
all_reviews = all_reviews.replace('!','')
all_reviews = all_reviews.replace('&',' ')
all_reviews = all_reviews.replace('/',' ')
all_reviews = all_reviews.replace('+',' ')
all_reviews = all_reviews.replace('‚','')
all_reviews = all_reviews.replace('”','')
all_reviews = all_reviews.replace('“','')

all_rev = all_reviews.split(' ')
all_words = sorted(list(set(all_rev)))
n_words = len(all_words)
print('Total Words: ', n_words)

char_to_int = dict((c, i+1) for i, c in enumerate(chars))
int_to_char = dict((i+1, c) for i, c in enumerate(chars))
char_to_int[''] = 0
int_to_char[0] = ''

word_to_int = dict((ch, j+1) for j, ch in enumerate(all_words))
int_to_word = dict((j+1, ch) for j, ch in enumerate(all_words))
word_to_int[''] = 0
int_to_word[0] = ''

int_reviews = []
for rev in reviews:
	rev = strip_punct(rev)
	words = rev.split(' ')
	int_reviews.append([word_to_int[w] for w in words])

char_reviews = []
for rev in reviews:
	char_reviews.append([char_to_int[char] for char in rev])

print('Original')
print(reviews[0])
print('Words')
print(int_reviews[0])
print('Chars')
print(char_reviews[0])

print('Original')
print(reviews[1])
print('Words')
print(int_reviews[1])
print('Chars')
print(char_reviews[1])

# Split into training and test set
ratio = 0.8
cutoff = int(len(int_reviews)*ratio)

reviews_train = int_reviews[0:cutoff]
reviews_test = int_reviews[cutoff+1:]
char_reviews_train = char_reviews[0:cutoff]
char_reviews_test = char_reviews[cutoff+1:]

prices_train = prices[0:cutoff]
prices_test = prices[cutoff+1:]

print('Number of training is ' ,len(reviews_train), ' = ', len(prices_train) )
print('Number of testing is ' ,len(reviews_test), ' = ', len(prices_test) )

print('Percentage expensive all: ',np.sum(prices)/len(prices))
print('Percentage expensive train: ',np.sum(prices_train)/len(prices_train))
print('Percentage expensive test: ',np.sum(prices_test)/len(prices_test))

pickle.dump(word_to_int,open('data/word_to_int.pkl','wb'))
pickle.dump(int_to_word,open('data/int_to_word.pkl','wb'))

pickle.dump(char_to_int,open('data/char_to_int.pkl','wb'))
pickle.dump(int_to_char,open('data/int_to_char.pkl','wb'))

pickle.dump(reviews,open('data/reviews.pkl','wb'))
pickle.dump(prices,open('data/prices.pkl','wb'))

pickle.dump(reviews_train,open('data/reviews_train.pkl','wb'))
pickle.dump(reviews_test,open('data/reviews_test.pkl','wb'))
pickle.dump(char_reviews_train,open('data/char_reviews_train.pkl','wb'))
pickle.dump(char_reviews_test,open('data/char_reviews_test.pkl','wb'))
pickle.dump(prices_train,open('data/prices_train.pkl','wb'))
pickle.dump(prices_test,open('data/prices_test.pkl','wb'))
