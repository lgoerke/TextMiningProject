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

words = pickle.load(open('data/word_freq_cheap.pkl','rb'))
a = words.keys()
b = words.values()

#c = sorted(words.items(), key=lambda x:x[1])
c = sorted(words.items(), key=lambda x:x[1], reverse = True)
#print(c[0:62])
#d = c[-62:-16]
d = c[0:16]
print(d)

# sort in-place from highest to lowest
d.sort(key=lambda x: x[1], reverse=True) 

# save the names and their respective scores separately
# reverse the tuples to go from most frequent to least frequent 
word_list = list(zip(*d))[0]
freq = list(zip(*d))[1]
x_pos = np.arange(len(word_list)) 
plt.barh(x_pos, freq,align='center')
plt.yticks(x_pos, word_list) 
plt.xlabel('Word frequency')
plt.title('Reviews for cheap wines')
plt.show()