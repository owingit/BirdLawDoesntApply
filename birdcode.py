import random
import os
import numpy as np
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier

#19 bird species
nclasses = 19

#makes the target array with dimensions numlabels, numclasses where 
#labels are encoded to have a 1
def makeTarget(labels):
	y = tuple([])
	for i in range(len(labels)):
		lbls_in_lbls = labels[i]
		y = y + tuple([lbls_in_lbls[1:len(lbls_in_lbls)]])

	y = labelsToBins(y)
	return y

def labelsToBins(lbls):
	num_labels = len(lbls)
	labels_in_bins = np.zeros(nclasses * num_labels)
	for i in range(num_labels):
		for j in range(len(lbls[i])):
			labels_in_bins[i * nclasses + lbls[i][j]] = 1

	return np.reshape(labels_in_bins, (num_labels, nclasses))
