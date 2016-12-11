from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import cluster
from sklearn import svm 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np
import random

import matplotlib.pyplot as plt
import csv as csv
from collections import defaultdict
import pandas as pd


#PROBLEM 2 

dataset = pd.read_csv('allPatients.csv',low_memory=False)
x = dataset.drop('Classes',axis=1)
x.replace(to_replace='?',value='NaN',inplace=True)
features = x.columns
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
x = imp.fit_transform(x)
x_imputed = pd.DataFrame(x,columns = features)
x_imputed.to_csv('allPatientsImputedtest.csv')
#features = x.columns
y = dataset['Classes']

clf = RandomForestClassifier(n_estimators=300, criterion = 'entropy', max_features=100)
clf.fit(x_imputed,y)



gene_freq = np.zeros(8560)
gene_freq = {}
for i in range(8560):
	gene_freq[i] = 0
for tree in clf.estimators_:
	i = 0
	for gene in tree.feature_importances_:
		if gene>0:
			gene_freq[i] = gene_freq[i]+1
		i = i+1

sortedgenes = sorted(gene_freq, key=gene_freq.get, reverse=True)
featuresp2 = sorted(list(clf.feature_importances_), reverse=True)
sortedtwohundred = sortedgenes[0:200]

#get top 200
top200 = []

column_name = list(x_imputed.columns.values)

for i in sortedtwohundred:
	top200.append(column_name[i])

#new dataset using chosen features
filtered = pd.read_csv('allPatients.csv',low_memory=False, usecols =top200)
filtered.replace(to_replace='?', value='NaN', inplace=True)

featuresp2_ = filtered.columns
imp1 = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
relevant_features = imp1.fit_transform(filtered)
relevant_genes = pd.DataFrame(relevant_features, columns=featuresp2_)

def getClusters(labels,data,k):
    d = {}
    for i in range(0,k):
        d["cluster"+str(i)] =  data.ix[np.nonzero(labels==i)]
    return d

def getDistance(clusters,distance):
    distDict = {}
    for cluster1 in itertools.combinations(clusters, 2):
        dist = cdist(clusters[cluster1[0]],clusters[cluster1[1]], distance)
        n,m = dist.shape
        distDict[cluster1[0]+" " +cluster1[1] + " Single Link"] = np.nanmin(dist)
        distDict[cluster1[0]+" " +cluster1[1] + " Complete Link"] = np.nanmax(dist)
        dist = np.nan_to_num(dist)
        distDict[cluster1[0]+" " + cluster1[1] + " Average"] = sum(sum(dist))/(n*m)
        distDict[cluster1[0] + " "+ cluster1[1] + " Centroid"] = np.amax(cdist(pd.DataFrame(clusters[cluster1[0]].mean()).transpose(),pd.DataFrame(clusters[cluster1[1]].mean()).transpose(),distance))
    return distDict



estimateEuclidean = {'euclidean_k2': KMeans(n_clusters=2).fit(relevant_genes),
              'euclidean_k3': KMeans(n_clusters=3).fit(relevant_genes),
              'euclidean_k4': KMeans(n_clusters=4).fit(relevant_genes)}

def new_euclidean_distance(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_similarity(X,Y)

#monkey patch for runtime attribute cosine similary measurements
KMeans.euclidean_distances = new_euclidean_distance

estimateDot = {'dot_k2': KMeans(n_clusters=2).fit(relevant_genes),
              'dot_k3': KMeans(n_clusters=3).fit(relevant_genes),
              'dot_k4': KMeans(n_clusters=4).fit(relevant_genes)}




for cluster in estimateEuclidean:
    k = len(np.unique(estimateEuclidean[cluster].labels_))
    groups = getClusters(estimateEuclidean[cluster].labels_,relevant_genes,len(np.unique(estimateEuclidean[cluster].labels_)))
    dist = getDistance(groups,'euclidean')
    pd.DataFrame([dist]).transpose().to_csv('ResultsEuclideanK({0}).csv'.format(k))
for cluster in estimateDot:
    k = len(np.unique(estimateDot[cluster].labels_))
    groups = getClusters(estimateDot[cluster].labels_,relevant_genes,k)
    dist = getDistance(groups,'cosine')
    pd.DataFrame([dist]).transpose().to_csv('ResultsDotK({0}).csv'.format(k))


#END PROBLEM 2

#BEGIN PROBLEM 3

#PROBLEM 3 PART 1

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x_imputed, y)

weights = clf.coef_[0]

coefs = {}
for i in range(8560):
	coefs[i] = 0
index = 0
for coef in clf.coef_:
	for wrap in coef:
		coefs[index] = abs(wrap)
		index = index+1


sortedgenesp3 = sorted(coefs, key=coefs.get, reverse=True)
sorted200p3 = sortedgenesp3[0:200]

top200p3 = []
columns = list(x_imputed.columns.values)
for coef in sorted200p3:
	top200p3.append(columns[coef])

print top200p3


#Problem 3 Part 2

#get best 100 features
sortedonehundred = sortedgenesp3[0:100]

top100 = []

for i in sortedonehundred:
	top100.append(column_name[i])

filteredp3 = pd.read_csv('allPatients.csv', low_memory=False, usecols = top100)
filteredp3.replace(to_replace='?', value='NaN', inplace=True)

features_p3_100 = filteredp3.columns
imp2 = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
relevant_featuresp3 = imp2.fit_transform(filteredp3)
relevant_genesp3 = pd.DataFrame(relevant_featuresp3, columns=features_p3_100)

#process and transform via quadratic kernel
poly = preprocessing.PolynomialFeatures(2)
poly_ = poly.fit_transform(relevant_genesp3)
featuresp3 = poly.get_feature_names(filteredp3.columns)

transformed = pd.DataFrame(poly_, columns = featuresp3)

#applying linear kernel to our transformation
clf1 = svm.SVC(kernel="linear")
clf1_p3 = clf1.fit(transformed,y)

poly_coefs = {}
for n in range(5151):
	poly_coefs[n] = 0
poly_index=0

for coeflist in clf1_p3.coef_:
	for coef in coeflist:
		poly_coefs[poly_index] = abs(coef)
		poly_index=poly_index+1

sorted_poly_coefs=sorted(poly_coefs, key=poly_coefs.get, reverse=True)
top200poly = sorted_poly_coefs[0:200]


top200p3_ = []

for i in top200poly:
	top200p3_.append(featuresp3[i])

print top200p3_
