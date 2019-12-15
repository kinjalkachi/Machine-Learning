#!/usr/bin/env python
# coding: utf-8

# In[82]:


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


# In[83]:


#Importing libraries: 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import itertools as it
from itertools import combinations

#Load dataset into pandas dataframe
df = pd.read_csv('/kaggle/input/claim_history.csv')
df1 = df


# In[85]:


#Data Wrangling as per need of further operations:
y = df1.CAR_USE
X = df1.drop('CAR_USE', axis=1)
df1.head()


# In[86]:


#Split the dataframe into training and testing data:
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=27513, stratify=df['CAR_USE'])
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)


# In[87]:


#a)	(5 points). 
#Please provide the frequency table (i.e., counts and proportions) of the target variable in the Training partition?

#Explanation:
#Frequency tables are a basic tool you can use to explore data and get an idea of the relationships between variables. 
#A frequency table is just a data table that shows the counts of one or more categorical variables. 
#One of the most useful aspects of frequency tables is that they allow you to extract the proportion of the data that 
#belongs to each category. With a one-way table, you can do this by dividing each table value by the total number 
#of records in the table:


# In[88]:


freq_y_train = pd.crosstab(index= y_train,  # Make a crosstab
                        columns="count", margins=True, margins_name="Total")      # Name the count column
freq_y_train["Proportions"] = (freq_y_train["count"]/len(y_train))*100

freq_y_train


# In[89]:


#b)	(5 points). 
#Please provide the frequency table (i.e., counts and proportions) of the target variable in the Test partition?


# In[90]:


freq_y_test = pd.crosstab(index= y_test,  # Make a crosstab
                        columns="count", margins=True, margins_name="Total")      # Name the count column
freq_y_test["Proportions"] = (freq_y_test["count"]/len(y_test))*100
freq_y_test


# In[91]:


#c)	(5 points). 
#What is the probability that an observation is in the Training partition given that CAR_USE = Commercial? 

#Prob(Train)
P_Train = 0.7
P_Test = 0.3

#Prob(Commercial|Training)
print(y_train[y_train=='Commercial'].count()/ y_train.shape[0])
P_Comm_train = y_train[y_train=='Commercial'].count()/ y_train.shape[0]

#Prob(Commercial|Test)
print(y_test[y_test=='Commercial'].count()/y_test.shape[0]) 
P_Comm_test = y_test[y_test=='Commercial'].count()/y_test.shape[0]

#Answer:
#Prob(Training|Commercial)
#Prob(Commercial|Training)*Prob(Train) / ((Prob(Commercial|Training)*Prob(Train)) + (Prob(Commercial|Test)*Prob(Test)))
Prob_Training_Commercial = P_Comm_train*P_Train / ((P_Comm_train*P_Train) + (P_Comm_test*P_Test))
print("Prob(Training|Commercial): ",Prob_Training_Commercial)

#Prob(Test|Commercial)
#Prob(Commercial|Test)*Prob(Test) / ((Prob(Commercial|Training)*Prob(Train)) + (Prob(Commercial|Test)*Prob(Test)))
Prob_Test_Commercial = P_Comm_test*P_Test / ((P_Comm_test*P_Train) + (P_Comm_test*P_Test))
print("Prob(Test|Commercial): ", Prob_Test_Commercial)


# In[92]:


#d)	(5 points). 
#What is the probability that an observation is in the Test partition given that CAR_USE = Private?

#Prob(Private|Training)
print(y_train[y_train=='Private'].count()/ y_train.shape[0])
P_Pri_train = y_train[y_train=='Private'].count()/ y_train.shape[0]

#Prob(Private|Test)
print(y_test[y_test=='Private'].count()/y_test.shape[0]) 
P_Pri_test = y_test[y_test=='Private'].count()/y_test.shape[0]

#Prob(Training|Private)
#Prob(Private|Training)*Prob(Train) / ((Prob(Private|Training)*Prob(Train)) + (Prob(Private|Test)*Prob(Test)))
Prob_Training_Pri = P_Pri_train*P_Train / ((P_Pri_train*P_Train) + (P_Pri_test*P_Test))
print("Prob(Training|Private): ",Prob_Training_Pri) 

#Answer:
#Prob(Test|Private)
#Prob(Private|Test)*Prob(Test) / ((Prob(Private|Training)*Prob(Train)) + (Prob(Private|Test)*Prob(Test)))
Prob_Test_Pri = P_Pri_test*P_Test / ((P_Pri_train*P_Train) + (P_Pri_test*P_Test))
print("Prob(Test|Private): ",Prob_Test_Pri)


# Question 2 (40 points)
# Please provide information about your decision tree.

# In[ ]:


#a)	(5 points). What is the entropy value of the root node?


# In[93]:


#Entropy of Root Node

commercial_cnt = df[df.CAR_USE == 'Commercial'].CAR_USE.count()
commercial_prob = commercial_cnt/len(df["CAR_USE"])

private_prob = (len(df["CAR_USE"])-commercial_cnt)/len(df["CAR_USE"])

Root_Entropy = -((commercial_prob * np.log2(commercial_prob) + private_prob * np.log2(private_prob)))
print("Entropy for root node is given as",Root_Entropy)


# In[94]:


#b)	(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?
#Answer: The split criteria on the variable Occupation with entropy and the categories:
#(Entropy: 0.7148805225259208, Category: ('Blue Collar', 'Unknown', 'Student'))


# In[95]:


dist_OCCUPATION = df.OCCUPATION.unique()
dist_CAR_TYPE = df.CAR_TYPE.unique()


# In[97]:


#Creating subsets of variables

def subset(variable):
    range_ =  round(len(variable)/2)+1
    if range_%2 == 0:
        subset_ls = []
        max_range = max(list(range(1,range_)))
        for i in range(1,range_):
            if i == max_range:
                temp = list(it.combinations(variable, i))
                for i in range(0,round(len(temp)/2)):
                    subset_ls.append(temp[i])
            else:
                temp = list(it.combinations(variable, i))
                for i in temp:
                    subset_ls.append(i)
        return(subset_ls)
    else:
        subset_ls = []
        for i in range(1,range_):
            temp = list(it.combinations(variable, i))
            for i in temp:
                subset_ls.append(i)
        return(subset_ls)
    
                 
Occupation_subsets = subset(dist_OCCUPATION)
Cartype_subsets = subset(dist_CAR_TYPE)


# In[98]:


#Verifying the number of subsets as per formula give in slide:
#(2^(k-1))-1
print("Number of Subsets of Occupation: ",len(Occupation_subsets))
print("Number of Subsets of Cartypes: ",len(Cartype_subsets))


# In[99]:


# Function to calculate table entropy for each variable:
import numpy
import pandas

# Define a function to visualize the percent of a particular target category by an interval predictor
def EntropyIntervalSplit (inData, split):         # input data frame (predictor in column 0 and target in column 1)
                                                  # split value

    dataTable = inData
    dataTable["Le_Split"] = False
    for ind in dataTable.index:
        if dataTable.iloc[:,0][ind] in split:
            dataTable["Le_Split"][ind] = True 
    
    crossTable = pandas.crosstab(index = dataTable['Le_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   

    nRows = crossTable.shape[0]
    nColumns = crossTable.shape[1]
   
    tableEntropy = 0
    for iRow in range(nRows-1):
        rowEntropy = 0
        for iColumn in range(nColumns):
            proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
            if (proportion > 0):
                rowEntropy -= proportion * numpy.log2(proportion)
        tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
    tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
    return tableEntropy 


# In[100]:


#Modify the x_train dataframe:
X_train["Labels"] = y_train


# In[102]:


#Making subsets of variable Education:
education_subset = [("Below High School",),("Below High School", "High School"),("Below High School", "High School","Bachelors"),("Below High School", "High School","Bachelors","Masters"),("Below High School", "High School","Bachelors","Masters","Doctors")]


# In[103]:


#Function to split the variable depending on entropy:
def split(indata, ls):
    ls_car_type = []
    for l in ls:
        #print(cs)
        EV = EntropyIntervalSplit(indata, list(l))
        ls_car_type.append((EV,l))
    return min(ls_car_type)


# In[105]:


#Passing parameters to create a split on variables:
indata_edu = X_train[["EDUCATION","Labels"]]
print(split(indata_edu, education_subset)) 

indata_type = X_train[["CAR_TYPE","Labels"]]
print(split(indata_type, Cartype_subsets))

indata_occ = X_train[["OCCUPATION","Labels"]]
print(split(indata_occ, Occupation_subsets))


# In[106]:


#Occupation variable has lowest entropy: (0.7148805225259208, ('Blue Collar', 'Unknown', 'Student'))
#Hence we divide the dataset with data pertaining to Occupation types:  'Blue Collar', 'Unknown', 'Student' 
#on left partition and remaining on right partition. 

Occupation_left = ['Blue Collar', 'Unknown', 'Student']

first_left = pd.concat([X_train[X_train.OCCUPATION == 'Blue Collar'], X_train[X_train.OCCUPATION == 'Unknown'], X_train[X_train.OCCUPATION == 'Student']])

first_right1 = pd.merge(X_train[X_train.OCCUPATION != 'Blue Collar'], X_train[X_train.OCCUPATION != 'Unknown'], how = 'inner')
first_right = pd.merge(first_right1, X_train[X_train.OCCUPATION != 'Student'], how = 'inner')

len(first_left), len(first_right)


# In[107]:


#Datacleaning for further use:
first_left = first_left[['OCCUPATION', 'CAR_TYPE', 'EDUCATION','Labels']]
first_right = first_right[['OCCUPATION', 'CAR_TYPE', 'EDUCATION','Labels']]
len(first_left), len(first_right)


# In[108]:


#Calculating Entropy to split data on Level2 left side:

indata_fl_edu = first_left[["EDUCATION","Labels"]]
print(split(indata_fl_edu, education_subset)) 

indata_fl_type = first_left[["CAR_TYPE","Labels"]]
print(split(indata_fl_type, Cartype_subsets))

indata_fl_occ = first_left[["OCCUPATION","Labels"]]
print(split(indata_fl_occ, Occupation_subsets))


# In[109]:


#Calculating Entropy to split data on Level2 right side:

indata_fr_edu = first_right[["EDUCATION","Labels"]]
print(split(indata_fr_edu, education_subset)) 

indata_fr_type = first_right[["CAR_TYPE","Labels"]]
print(split(indata_fr_type, Cartype_subsets))

indata_fr_occ = first_right[["OCCUPATION","Labels"]]
print(split(indata_fr_occ, Occupation_subsets))


# In[110]:


Education_left = ['Below High School']

second_left_left = first_left[first_left.EDUCATION == 'Below High School']
second_left_right = first_left[first_left.EDUCATION != 'Below High School']


# In[111]:


Cartype_right = ['Minivan', 'SUV', 'Sports Car']
second_right_left = pd.concat([first_right[first_right.CAR_TYPE == 'Minivan'], first_right[first_right.CAR_TYPE == 'SUV'], 
                               first_right[first_right.CAR_TYPE == 'Sports Car']])

second_right_right = first_right[(first_right["CAR_TYPE"] != "Minivan") & (first_right["CAR_TYPE"] != "Sports Car") & 
                              (first_right["CAR_TYPE"] != "SUV")]


# In[112]:


#e)	(15 points). Describe all your leaves.  Please include the decision rules and the counts of the target values.
#Answer: 

print("Decision rules are as follows: ")
print("Confidence of occurance of Event:'Commercial'at leaf nodes")
print(second_left_left[second_left_left.Labels == 'Commercial'].Labels.count()/second_left_left.Labels.count())
print(second_left_right[second_left_right.Labels == 'Commercial'].Labels.count()/second_left_right.Labels.count())
print(second_right_left[second_right_left.Labels == 'Commercial'].Labels.count()/second_right_left.Labels.count())
print(second_right_right[second_right_right.Labels == 'Commercial'].Labels.count()/second_right_right.Labels.count()) 

print("Count of Target values on all the leaf nodes are as follows: ")
print(second_left_left[second_left_left.Labels == 'Commercial'].Labels.count())
print(second_left_right[second_left_right.Labels == 'Commercial'].Labels.count())
print(second_right_left[second_right_left.Labels == 'Commercial'].Labels.count())
print(second_right_right[second_right_right.Labels == 'Commercial'].Labels.count()) 


# In[113]:


#Please apply your decision tree to the Test partition and then provide the following information.
#a)	(10 points). Use the proportion of target Event value in the training partition as the threshold, 
#what is the Misclassification Rate in the Test partition?

threshold = X_train[X_train.Labels == "Commercial"].Labels.count()/X_train.ID.count()
threshold


# In[114]:


#Assigning probablities to test dataframe:
x_test = X_test[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]

occ = ['Blue Collar', 'Unknown', 'Student']
car = ['Minivan', 'SUV', 'Sports Car']
ed = ['Below High School']

pred_prob = []
x_test["pred_prob"] = ""

for ind in x_test.index:
    if x_test.iloc[:,1][ind] in occ:
        if x_test.iloc[:,2][ind] in ed:
            pred_prob.append(0.24647887323943662) 
            x_test["pred_prob"][ind] = 0.24647887323943662
        else:
            pred_prob.append(0.8504761904761905)   
            x_test["pred_prob"][ind] = 0.8504761904761905
    else:
        if x_test.iloc[:,0][ind] in car:
            pred_prob.append(0.006151953245155337)
            x_test["pred_prob"][ind] = 0.006151953245155337
        else:
            pred_prob.append(0.5464396284829721)
            x_test["pred_prob"][ind] = 0.5464396284829721


# In[115]:


#Assigning the labels to the test data as per the calculated threshold:
x_test["Labels"] = ""
for ind in x_test.index:
    if x_test["pred_prob"][ind] >= threshold:
        x_test["Labels"][ind]= "Commercial"
    else:
        x_test["Labels"][ind]= "Private"


# In[116]:


#Test dataframe with predicted probablities and predicted labels:
x_test.head()


# In[117]:


#Q3)a a)	(10 points). Use the proportion of target Event value in the training partition as the threshold, 
#what is the Misclassification Rate in the Test partition?
from sklearn.metrics import accuracy_score
print("Missclassification Rate",1-accuracy_score(y_test,x_test["Labels"]))


# In[118]:


#b)	(10 points). What is the Root Average Squared Error in the Test partition? 
import math
# Calculate the Root Average Squared Error
RASE = 0.0
for i in range (0,len(y_test)):
    if y_test.iloc[i] == "Commercial":
        RASE += (1-pred_prob[i])**2
    else:
        RASE += (pred_prob[i])**2
RASE = math.sqrt(RASE/len(y_test))
RASE


# In[119]:


#c)	(10 points). What is the Area Under Curve in the Test partition? 

# Calculate the Root Mean Squared Error

from sklearn import metrics
true_values = 1.0*np.isin(y_test,"Commercial")
RMSE = metrics.mean_squared_error(true_values, pred_prob)
RMSE = np.sqrt(RMSE)
RMSE 

AUC = metrics.roc_auc_score(true_values, pred_prob)
AUC

print("Root Average Squared Error",RASE)
print("Root Mean Squared Error",RMSE)
print("Area Under the curve",AUC)
print("Missclassification Rate",1-accuracy_score(y_test,x_test["Labels"]))


# In[120]:


OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(y_test, pred_prob, pos_label = 'Commercial')


# In[121]:


OneMinusSpecificity = np.append([0], OneMinusSpecificity)
Sensitivity = np.append([0], Sensitivity)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

print(Sensitivity)
print(Sensitivity)


# In[122]:


#d)	(10 points). Generate the Receiver Operating Characteristic curve for the Test partition.  
#The axes must be properly labeled.  Also, don’t forget the diagonal reference line. 

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

