#!/usr/bin/env python
# coding: utf-8

# In[133]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# a)	(10 points) Create a dataset which contains the number of distinct items in each customer’s market basket. Draw a histogram of the number of unique items.
#What are the median, the 25th percentile and the 75th percentile in this histogram?

# In[134]:


#importing the libraries:
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[135]:



#Import dataset:
df = pd.read_csv('/kaggle/input/Groceries.csv', delimiter=',') 

#Finding total number of unique customers:
total_cust = len(df["Customer"].unique()) 

#Creating dataset which contains number of distinct items in each customer’s market basket:
ListItem = df.groupby(['Customer'])['Item'].apply(list).values.tolist() 

#Use ItemIndicator method to find desired dataset:
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)

#Final dataset which gives distint items in each customer’s market basket:
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
nItemPurchase = df.groupby('Customer').size()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
nItemPurchase = df.groupby('Customer').size()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
final = pd.DataFrame(freqTable.index.values.astype(int), columns = ['Unique_Item_set'])
final['Number of customer per unique item'] = freqTable.values.astype(int)
print("Dataset for distinct items in each customer's market basket: \n",final)


# In[136]:


#Histogram of number of unique items
ls = []
for i in ListItem:
    ls.append(len(i))
plt.hist(ls, bins = 32)
plt.xlabel("Unique Item sets")
plt.ylabel("Number of customer per unique item")


# In[137]:


#The median, the 25th percentile and the 75th percentile in this histogram are:
hist_df = pd.DataFrame(ls, columns = ['Customers per unique item'])
hist_df.describe() 

#Answer: The median is 3. 25th percentile of dataset is : 2. 75th percentile is 6.


# b)	(10 points) If you are interested in the k-itemsets which can be found in the market baskets of at least seventy five (75) customers.  How many itemsets can you find?  Also, what is the largest k value among your itemsets?

# In[138]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 

frequent_itemsets = apriori(ItemIndicator, min_support = 75/total_cust, max_len = 32, use_colnames = True)
print("Number of itemsets are: ",frequent_itemsets.count()[1])
print("Largest k value among the itemsets is: ",len(frequent_itemsets['itemsets'].apply(list).values.tolist()[-1]))


# c)	(10 points) Find out the association rules whose Confidence metrics are at least 1%.  How many association rules have you found?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.  Also, you do not need to show those rules.

# In[139]:


# Association rules for confidence matrix at atleast 1%
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("Number of association rules are: ", assoc_rules.count()[1])


# d)	(10 points) Graph the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you found in (c).  Please use the Lift metrics to indicate the size of the marker

# In[140]:


#Plot for Confidence versus vs. Support matrix using Lift matrix to indicate the size of the marker.
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()


# e)	(10 points) List the rules whose Confidence metrics are at least 60%.  Please include their Support and Lift metrics.

# In[141]:


# Association rules for confidence matrix at atleast 60%
assoc_rules1 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print("Association rules for 60% confidence: \n"),
assoc_rules1


# Question 2 (50 points)
# #Apply the Spectral Clustering method to the Spiral.csv.  Your input fields are x and y. Wherever needed, specify random_state = 60616 in calling the KMeans function.
# 

# 2a)	(10 points) Generate a scatterplot of y (vertical axis) versus x (horizontal axis).  How many clusters will you say by visual inspection?

# In[142]:


Spiral = pd.read_csv('/kaggle/input/Spiral.csv', delimiter=',')
plt.scatter(Spiral['x'], Spiral['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#I can see 2 groups after visualizing the graph.


# 2b)	(10 points) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[143]:


import sklearn.cluster as cluster
trainData = Spiral[['x','y']]

#From above scatter plot I see that there are 2 clusters, hence I have n_clusters=2
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)
print("Cluster Centroids = \n", kmeans.cluster_centers_)
Spiral['KMeanCluster'] = kmeans.labels_ 

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[144]:



import numpy
import sklearn 
import math

nObs = Spiral.shape[0]

# I got best solution as three nearest neighbors
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = numpy.zeros((nObs, nObs))
Degree = numpy.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(numpy.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()


# 2c)	(10 points) Apply the nearest neighbor algorithm using the Euclidean distance.  How many nearest neighbors will you use?  Remember that you may need to try a couple of values first and use the eigenvalue plot to validate your choice.

# 2d)Retrieve the first two eigenvectors that correspond to the first two smallest eigenvalues.  Display up to ten decimal places the means and the standard deviation of these two eigenvectors.  Also, plot the first eigenvector on the horizontal axis and the second eigenvector on the vertical axis.

# In[145]:



from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)
# Values of the eigenvectors 
Z = evecs[:,[0,1]]

print("Frist two Eigenvectors are\n",Z[[0]],"\n", Z[[1]])
print("Mean of first Eigenvectors",round(Z[[0]].mean(),10))
print("Standard deviation of first Eigenvectors",round(Z[[0]].std(),10))
print("Mean of second Eigenvectors",round(Z[[1]].mean(),10))
print("Standard deviation of first Eigenvectors",round(Z[[1]].std(),10))

#Plotting Scatter of eigen vectore
plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()


# 2e)	(10 points) Apply the K-mean algorithm on your first two eigenvectors that correspond to the first two smallest eigenvalues.
#Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[146]:


#Apply K-means clustering to eigen-vectors to get good clustering of spiral data.
kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)
Spiral['SpectralCluster'] = kmeans_spectral.labels_
plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[ ]:




