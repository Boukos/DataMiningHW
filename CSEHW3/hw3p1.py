from sklearn import preprocessing
from sklearn import cluster
from scipy.stats import pearsonr
from fancyimpute import KNN
from sklearn.cluster import KMeans
import itertools
import numpy as np
import random
import csv as csv
import pandas as pd

#PROBLEM 1.1 IMPUTE USING 7 NEAREST NEIGHBORS

dataset = pd.read_csv('ctrl1.csv',low_memory=False)
dataset_case = pd.read_csv('case2.csv',low_memory=False)
x = dataset.drop('Classes',axis=1)
x_case = dataset.drop('Classes',axis=1)
x_case.replace(to_replace='?',value='NaN',inplace=True)
x.replace(to_replace='?',value='NaN',inplace=True)
features = x.columns
features_case = x_case.columns

X_filled_knn_ctrl = KNN(k=7).complete(x)
X_filled_knn_case = KNN(k=7).complete(x_case)
x_imputed = pd.DataFrame(X_filled_knn, columns=features)
x_imputed_case = pd.DataFrame(X_filled_knn_case, columns=features)
x_imputed.to_csv('knn7control.csv')


#PROBLEM 1.2 BOTTOM UP CLUSTERING
#imputing combined patient dataset
dataset_all = pd.read_csv('allPatients.csv',low_memory=False)
x_all = dataset_all.drop('Classes',axis=1)
x_all.replace(to_replace='?',value='NaN',inplace=True)
features_all = x_all.columns
X_filled_knn_all = KNN(k=7).complete(x_all)
x_imputed_all = pd.DataFrame(X_filled_knn_all, columns=features_all)
#x_imputed_all.to_csv('allPatientsKnn7.csv')
y = dataset_all['Classes']

#using cosine similarity centered with means subtracted (pearson correlation)
dataset_scale=preprocessing.scale(x_imputed_all) 

#performing agglomerative clustering on imputed dataset

bucluster = cluster.AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage = 'complete')
#set above to n_clusters=8 and 4 and 2 to compare with third and second level of top down clustering
bucluster.fit(dataset_scale)

clusters = {k: [] for k in np.unique(bucluster.labels_)}
for i, v in enumerate(bucluster.labels_):
	clusters[v].append(i+1)
print clusters



#PROBLEM 1.3 TOP DOWN CLUSTERING 

# (RECURSIVE K-MEANS)
# def topdown(dataset):

# 	if len(dataset)<2:
# 		return dataset.iloc[[0]]

# 	kmeansclusters = KMeans(n_clusters=2).fit(dataset)

#	df1, df2 = getClusters(kmeanclusters.labels_)

# 	Kmeans(n_clusters=2).fit(df1)
# 	Kmeans(n_clusters=2).fit(df2)

# 	return topdown(df1)
#	return topdown(df2)


def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
	return pearsonr(X,Y)

#monkey patch for pearson correlation
KMeans.euclidean_distances = new_euclidean_distances
print len(x_imputed_all)

#level 0
kmeansclusters_0 = KMeans(n_clusters=2).fit(x_imputed_all)

cluster1 = np.where(kmeansclusters_0.labels_ ==1)[0]
cluster2 = np.where(kmeansclusters_0.labels_ ==0)[0]
print len(kmeansclusters_0.labels_)


df1 = x_imputed_all.iloc[cluster1]
df2 = x_imputed_all.iloc[cluster2]

#cluster comparison to bottom up
clusters_td = {k: [] for k in np.unique(kmeansclusters_0.labels_)}
for i, v in enumerate(kmeansclusters_0.labels_):
	clusters_td[v].append(i+1)
print clusters_td
sim = set(df1_bu.index) & set(df1.index) 
print len(sim)
print set(df1_bu.index)-set(df1.index)
print set(df1.index)-set(df1_bu.index)

#level 2
kmeansclusters_1_1 = KMeans(n_clusters=2).fit(df1)
kmeansclusters_1_2 = KMeans(n_clusters=2).fit(df2)

cluster1_2 = np.where(kmeansclusters_1_1.labels_ ==1)[0]
cluster2_2 = np.where(kmeansclusters_1_1.labels_ ==0)[0]
cluster3_2 = np.where(kmeansclusters_1_2.labels_ ==1)[0]
cluster4_2 = np.where(kmeansclusters_1_2.labels_ ==0)[0]

df1_2 = df1.iloc[cluster1_2]
df2_2 = df1.iloc[cluster2_2]
df3_2 = df2.iloc[cluster3_2]
df4_2 = df2.iloc[cluster4_2]

print len(df1_2)
print len(df2_2)
print len(df3_2)
print len(df4_2)
print df1_2.index
print df2_2.index
print df3_2.index
print df4_2.index

#level 3
kmeansclusters_2_1 = KMeans(n_clusters=2).fit(df1_2)
kmeansclusters_2_2 = KMeans(n_clusters=2).fit(df2_2)
kmeansclusters_2_3 = KMeans(n_clusters=2).fit(df3_2)
kmeansclusters_2_4 = KMeans(n_clusters=2).fit(df4_2)

cluster1_3 = np.where(kmeansclusters_2_1.labels_ ==1)[0]
cluster2_3 = np.where(kmeansclusters_2_1.labels_ ==0)[0]
cluster3_3 = np.where(kmeansclusters_2_2.labels_ ==1)[0]
cluster4_3 = np.where(kmeansclusters_2_2.labels_ ==0)[0]
cluster5_3 = np.where(kmeansclusters_2_3.labels_ ==1)[0]
cluster6_3 = np.where(kmeansclusters_2_3.labels_ ==0)[0]
cluster7_3 = np.where(kmeansclusters_2_4.labels_ ==1)[0]
cluster8_3 = np.where(kmeansclusters_2_4.labels_ ==0)[0]

df1_3 = df1_2.iloc[cluster1_3]
df2_3 = df1_2.iloc[cluster2_3]
df3_3 = df2_2.iloc[cluster3_3]
df4_3 = df2_2.iloc[cluster4_3]
df5_3 = df3_2.iloc[cluster5_3]
df6_3 = df3_2.iloc[cluster6_3]
df7_3 = df4_2.iloc[cluster7_3]
df8_3 = df4_2.iloc[cluster8_3]

print len(df1_3)
print len(df2_3)
print len(df3_3)
print len(df4_3)
print len(df5_3)
print len(df6_3)
print len(df7_3)
print len(df8_3)
print df1_3.index
print df2_3.index
print df3_3.index
print df4_3.index
print df5_3.index
print df6_3.index
print df7_3.index
print df8_3.index


