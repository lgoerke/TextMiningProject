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
from sklearn.decomposition import PCA

def mkdir_p(path):
	if not os.path.exists(path):
		os.makedirs(path)

def cluster(print_number):
	
	seed = 4715

	i = datetime.now()
	orig_stdout = sys.stdout
	orig_stderr = sys.stderr
	logname = 'log_cluster06_' + print_number +'.txt'
	f = open(logname, 'w')
	sys.stdout = f
	sys.stderr = f


	print('---')
	print('Loading')
	print('---')

	filepath = 'data/reviews_categorical01_n.pkl'
	a = pickle.load(open(filepath,'rb'))
	print(len(a))
	print(len(a[0]))
	print(len(a[0][0]))

	filepath = 'data/reviews_categorical02_n.pkl'
	b = pickle.load(open(filepath,'rb'))

	filepath = 'data/reviews_categorical03_n.pkl'
	c = pickle.load(open(filepath,'rb'))

	filepath = 'data/reviews_categorical04_n.pkl'
	d = pickle.load(open(filepath,'rb'))

	filepath = 'data/reviews_categorical05_n.pkl'
	e = pickle.load(open(filepath,'rb'))

	all_reviews = a + b + c + d + e
	for idx, entry in enumerate(all_reviews):
		all_reviews[idx] = all_reviews[idx][0]


	# print('---')
	# print('Loading')
	# print('---')

	# all_reviews = pickle.load(open('data/reviews_categorical.pkl','rb'))

	print('---')
	print('Done Loading')
	print('---')

	#all_reviews = np.array(all_reviews)

	#print(all_reviews.shape)
	#print(type(all_reviews))

	#all_reviews = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]

	print(len(all_reviews))
	print(len(all_reviews[0]))
	print(len(all_reviews[1]))
	print(len(all_reviews[2]))
	print(all_reviews[0])
	print(all_reviews[1])
	print(all_reviews[2])

	print('---')
	print('PCA')
	print('---')

	pca = PCA(n_components=3)
	transformed_reviews = pca.fit_transform(all_reviews)

	filepath = 'data/transformed_reviews_3d.pkl'
	pickle.dump(transformed_reviews,open(filepath,'wb'))

	print('---')
	print('Done PCA')
	print('---')

	sys.stderr = orig_stderr
	sys.stdout = orig_stdout
	f.close()




def printUsage():
	print("Usage: 05_cluster.py {Number of run}")
def main(args):
	if not(len(args) == 1):
		printUsage()
	else:
		print_number = args[0]
		cluster(print_number)
 
if __name__ == "__main__":
	main(sys.argv[1:])
