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

orig_stdout = sys.stdout
orig_stderr = sys.stderr
logname = 'word_freq_cheap.txt'
f = open(logname, 'w')
sys.stdout = f
sys.stderr = f

reviews = pickle.load(open('data/reviews.pkl','rb'))
prices = pickle.load(open('data/prices.pkl','rb'))

reviews_train = pickle.load(open('data/reviews_train.pkl','rb'))
reviews_test = pickle.load(open('data/reviews_test.pkl','rb'))
prices_train = pickle.load(open('data/prices_train.pkl','rb'))
prices_test = pickle.load(open('data/prices_test.pkl','rb'))

#all_reviews = reviews_train + reviews_test
#all_reviews = [item for sublist in all_reviews for item in sublist]
all_prices = prices_train + prices_test
#no_all = len(all_reviews)

for idx, entry in enumerate(reviews):
	reviews[idx] = reviews[idx].replace('.','')
	reviews[idx]  = reviews[idx].replace(',','')
	reviews[idx]  = reviews[idx].replace(':','')
	reviews[idx]  = reviews[idx].replace(';','')
	reviews[idx]  = reviews[idx].replace('"','')
	reviews[idx]  = reviews[idx].replace(')','')
	reviews[idx]  = reviews[idx].replace('(','')
	reviews[idx]  = reviews[idx].replace('#','')
	reviews[idx]  = reviews[idx].replace('?','')
	reviews[idx]  = reviews[idx].replace('!','')
	reviews[idx]  = reviews[idx].replace('&',' ')
	reviews[idx]  = reviews[idx].replace('/',' ')
	reviews[idx]  = reviews[idx].replace('+',' ')

print(len(reviews))
print(len(all_prices))

reviews_exp = []
reviews_cheap = []
for idx,rev in enumerate(reviews[:-1]):
	if (all_prices[idx] == 1):
		reviews_exp.append(rev)
	else:
		reviews_cheap.append(rev)

no_all = len(reviews_cheap)
print(no_all)
#print(reviews_cheap)
all_rev = ' '.join(reviews_cheap)
all_rev = all_rev.split(' ')
#print(all_rev)
print('yo')
print(Counter(all_rev))

pickle.dump(Counter(all_rev),open('data/word_freq_cheap.pkl','wb'))
print('yo')

sys.stderr = orig_stderr
sys.stdout = orig_stdout
f.close()