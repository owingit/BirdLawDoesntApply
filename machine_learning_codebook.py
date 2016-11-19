
# coding: utf-8

# In[7]:

import numpy as np
from sklearn import cluster
from scipy.cluster.vq import whiten 

k = 50 
kextra = 10 
num_recs = 645
seed = 2

segment_file = open('bird_data/supplemental_data/segment_features.txt',
               'r')
               
##clean
line = segment_file.readline()
line = segment_file.readline()
index = 0
while line != '':
    tokens = line.split(',')
    nums = map(float, tokens) 
    nums = nums[2:len(line)] # Omit recid and segid
    
    if index == 0:
        segfeatures = nums
    else:
        segfeatures = np.vstack((segfeatures, nums))

    line = segment_file.readline()
    index += 1
    
#Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening. 
#Each feature is divided by its standard deviation across all observations to give it unit variance.
#From documentation in scikit-learn

segfeatures = whiten(segfeatures)


# In[8]:

segfeatures


# In[14]:

kmeans1 = cluster.KMeans(n_clusters=k, init='k-means++', n_init=k,
                         max_iter=300, random_state=seed)
kmeans2 = cluster.KMeans(n_clusters=kextra, init='k-means++', n_init=k,
                         max_iter=300, random_state=seed)

clusters1 = kmeans1.fit_predict(segfeatures)
clusters2 = kmeans2.fit_predict(segfeatures)
segment_file = open('bird_data/supplemental_data/segment_features.txt',
               'r')
segment_file.seek(0)
line = segment_file.readline()
line = segment_file.readline()
index = 0
prev_rec_id = -1
hist = np.zeros((num_recs, k + kextra))
while line != '':
    while 1:
        tokens = line.split(',')
        rec_id = int(tokens[0]) 
        if rec_id != prev_rec_id:
            prev_rec_id = rec_id
            break

        hist[rec_id][clusters1[index]] += 1
        hist[rec_id][k + clusters2[index]] += 1

        line = segment_file.readline()
        if line == '':
            break

        index += 1

segment_file.close()

histfilename = 'hist.txt'
histfile = open(histfilename, 'w')
histfile.write('rec_id,[hist]\n')

for rec_id in range(num_recs):
    histfile.write('%d,' % rec_id)   
    for col in range(k + kextra - 1):
        histfile.write('%f,' % hist[rec_id][col])   

    histfile.write('%f\n' % hist[rec_id][col + 1])   
histfile.close()


# In[ ]:



