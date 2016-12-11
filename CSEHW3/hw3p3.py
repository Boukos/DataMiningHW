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


for i in range(2,21):
	
	svd = decomposition.TruncatedSVD(n_components=i)
	transformed = svd.fit_transform(df_scaled)
	dfTransformed = pd.DataFrame(transformed)

	def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
		return pearsonr(X,Y)

	KMeans.euclidean_distances = new_euclidean_distances

	kmeansSVD = KMeans(n_clusters=2).fit(dfTransformed)

	print "k="+str(i)
	print kmeansSVD.labels_
	cluster1 = np.where(kmeansSVD.labels_ ==1)[0]
	cluster2 = np.where(kmeansSVD.labels_ ==0)[0]
	df1 = dataset.iloc[cluster1]
	df2 = dataset.iloc[cluster2]
	print df1, df2
