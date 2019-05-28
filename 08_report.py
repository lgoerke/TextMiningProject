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

predictions = pickle.load(open('data/predicted_classes_l_007.pkl','rb'))
prices = pickle.load(open('data/prices_test.pkl','rb'))

pred = []
for p in predictions:
	if (p[0] < 0.5):
		pred.append(0)
	else:
		pred.append(1)

#print(pred)
#print(prices)

report = classification_report(prices, pred)
print(report)

sum_true_expensive = 0
sum_true_cheap = 0
recog_expensive = 0
recog_cheap = 0
true_expensive = 0
true_cheap = 0
for idx,p in enumerate(prices):
	if ((p == pred[idx]) and (p == 1)):
		sum_true_expensive = sum_true_expensive + 1
	elif ((p == pred[idx]) and (p == 0)):
		sum_true_cheap = sum_true_cheap + 1
	if (p == 1):
		true_expensive = true_expensive + 1
	elif (p == 0):
		true_cheap = true_cheap + 1
	if (pred[idx] == 1):
		recog_expensive = recog_expensive + 1
	elif (pred[idx] == 0):
		recog_cheap = recog_cheap + 1

print('SUM: ',true_cheap+true_expensive,' = ',recog_cheap+recog_expensive)
print('True (1): ', true_expensive)
print('True (0): ', true_cheap)
print('Found (1): ', recog_expensive)
print('Found (0): ', recog_cheap)

print('Precision (1): ', sum_true_expensive/recog_expensive)
print('Precision (0): ', sum_true_cheap/recog_cheap)
print('Recall (1): ', sum_true_expensive/true_expensive)
print('Recall (0): ', sum_true_cheap/true_cheap)
