from sklearn import svm 
from sklearn import preprocessing
import numpy as np
#from numpy import genfromtxt, savetxt
import matplotlib.pyplot as plt
import csv as csv
import pandas as pd


dataset = pd.read_csv('allPatients_top100.csv',low_memory=False)
# x = dataset.drop('Classes',axis=1)
# x.replace(to_replace='?',value='NaN',inplace=True)
# features = x.columns
# imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
# x = imp.fit_transform(x)
# x_imputed = pd.DataFrame(x,columns = features)
#features = x.columns
#y = dataset['Classes']
#dataset = csv.reader(open('allPatients.csv','rU'))
#dataset = pd.read_csv('allPatients.csv',low_memory=False)
#x = dataset.drop('Classes',axis=1)
#x.replace(to_replace='?',value=0,inplace=True)
#features = x.columns
#y = dataset['Classes']
feature_names = dataset.columns

poly = preprocessing.PolynomialFeatures(2)
poly_ = poly.fit_transform(dataset)
featuresp3 = poly.get_feature_names(feature_names)
#print featuresp3
transformed = pd.DataFrame(poly_, columns = featuresp3)
print "here"
top200p3 = []

for i in sortedtwohundred:
	top100p3.append(featuresp3[i])

print top200p3