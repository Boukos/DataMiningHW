from __future__ import division
from sklearn import preprocessing
from sklearn import cluster
from sklearn import decomposition
from fancyimpute import KNN
from sklearn.cluster import KMeans
import itertools
import numpy as np
import random
import csv as csv
import pandas as pd


dataset = pd.read_csv('allPatientsKnn7.csv',low_memory=False)

df_scaled = preprocessing.scale(dataset) 
n_samples = dataset.shape[0]
print dataset.shape[0]


#Part 1 - Get Eigenvectors. Algorithm returns n = # of samples worth of eigenvalues if unspecified
pca1 = decomposition.PCA()
pca1.fit(df_scaled)

a = np.asarray(pca1.explained_variance_)
np.savetxt("eigenvalues.csv", a, delimiter =",")

#Part 2 - Specifying k=2 to 20 eigenvectors to use
for i in range(2,21):

	pca = decomposition.PCA(n_components=i)
	pca.fit(df_scaled)

	dataDecomposed = pca.fit_transform(df_scaled)
	dfDecomposed = pd.DataFrame(dataDecomposed)

	def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
		return pearsonr(X,Y)

	KMeans.euclidean_distances = new_euclidean_distances

	kmeansPCA = KMeans(n_clusters=2).fit(dfDecomposed)

	print "k=" + str(i)
	print kmeansPCA.labels_
	cluster1 = np.where(kmeansPCA.labels_ ==1)[0]
	cluster2 = np.where(kmeansPCA.labels_ ==0)[0]
	df1 = dataset.iloc[cluster1]
	df2 = dataset.iloc[cluster2]
	print df1, df2
	





