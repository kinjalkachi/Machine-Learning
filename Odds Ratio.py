#!/usr/bin/env python
# coding: utf-8

# # Question 2

# In[4]:


#Import required libraries and dataset
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as stats

dataframe = pd.read_csv("C:/Users/kinja/Downloads/Purchase_Likelihood.csv")


# In[8]:


#a) (5 points) Show in a table the frequency counts and the Class Probabilities of the target variable.
# Calculate the probabilities of each class
frequency = dataframe.groupby('A').size()
table = pd.DataFrame(columns = ['Count','Class_probability'])
table.Count = frequency
table.Class_probability = table.Count/dataframe.shape[0]
print(table)


# In[9]:


#b) (5 points) Show the crosstabulation table of the target variable by the feature group_size.  The table contains the frequency counts.
gs_crosstab = pd.crosstab(df.A,df.group_size)
gs_crosstab


# In[12]:


#c)  Show the crosstabulation table of the target variable by the feature homeowner.  The table contains the frequency counts.
ho_crosstab = pd.crosstab(df.A,df.homeowner)
ho_crosstab


# In[11]:


#d) Show the crosstabulation table of the target variable by the feature married_couple.  The table contains the frequency counts.
mc_crosstab = pd.crosstab(df.A,df.married_couple)
mc_crosstab


# In[17]:


# Function to calculate cramers v statistic
import scipy.stats as ss
def cramers_v_statistic(confusion_matrix):
    chi_squared = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi_2 = chi_squared/n
    r,k = confusion_matrix.shape
    phi2corr = max(0,(phi_2 - ((k-1)*(r-1))/(n-1)))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    print(np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1))))


# In[36]:


#e)(10 points) Calculate the Cramer’s V statistics for the above three crosstabulations tables.  
#Based on these Cramer’s V statistics, which feature has the largest association with the target A?

print("The Cramers V Statistic values for each variable are as follows \n")
print("For group_size")
print(cramers_v_statistic(gs_crosstab))
print()

print("For homeowner")
print(cramers_v_statistic(ho_crosstab))
print()

print("For married_couple")
print(cramers_v_statistic(mc_crosstab))
print()


# In[79]:


#g)(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for A = 0, 1, 2 based on the Naïve Bayes model. 
#List your answers in a table with proper labelling.
group_sizes = sorted(list(dataframe.group_size.unique()))
homeowners = sorted(list(dataframe.homeowner.unique()))
married_couples = sorted(list(dataframe.married_couple.unique()))

data_group_size = pd.crosstab(df.A, df.group_size, margins = False, dropna = False)
data_homeowner = pd.crosstab(df.A, df.homeowner, margins = False, dropna = False)
data_married_couple = pd.crosstab(df.A, df.married_couple, margins = False, dropna = False)


# In[65]:


group_sizes = sorted(list(df.group_size.unique()))
homeowners = sorted(list(df.homeowner.unique()))
married_couples = sorted(list(df.married_couple.unique()))


# In[63]:


import itertools
all_combi = list(itertools.product(group_sizes, homeowners, married_couples))


# In[64]:


def get_valid_probabilities(predictors):
    cond_prob_0 = ((data_grouped['Count'][0] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][0] / data_group_size.loc[[0]].sum(axis=1)[0]) * 
                   (data_homeowner[predictors[1]][0] / data_homeowner.loc[[0]].sum(axis=1)[0]) * 
                   (data_married_couple[predictors[2]][0] / data_married_couple.loc[[0]].sum(axis=1)[0]))
    cond_prob_1 = ((data_grouped['Count'][1] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][1] / data_group_size.loc[[1]].sum(axis=1)[1]) * 
                   (data_homeowner[predictors[1]][1] / data_homeowner.loc[[1]].sum(axis=1)[1]) * 
                   (data_married_couple[predictors[2]][1] / data_married_couple.loc[[1]].sum(axis=1)[1]))
    cond_prob_2 = ((data_grouped['Count'][2] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][2] / data_group_size.loc[[2]].sum(axis=1)[2]) * 
                   (data_homeowner[predictors[1]][2] / data_homeowner.loc[[2]].sum(axis=1)[2]) * 
                   (data_married_couple[predictors[2]][2] / data_married_couple.loc[[2]].sum(axis=1)[2]))
    sum_cond_probs = cond_prob_0 + cond_prob_1 + cond_prob_2
    valid_prob_0 = cond_prob_0 / sum_cond_probs
    valid_prob_1 = cond_prob_1 / sum_cond_probs
    valid_prob_2 = cond_prob_2 / sum_cond_probs

    return [valid_prob_0, valid_prob_1, valid_prob_2]


# In[67]:


predicted_nb_probabilities = []
for combination in all_combi:
    temp = [get_valid_probabilities(combination)]
    predicted_nb_probabilities.extend(temp)
df_nb_predicted_prbabilities = pd.DataFrame(predicted_nb_probabilities,columns = ["0","1","2"])
df_nb_predicted_prbabilities


# In[81]:


combi_df = pd.DataFrame(np.array(all_combi),columns = ["group","home","married"])
pd.DataFrame.join(combi_df,df_nb_predicted_prbabilities)


# In[75]:


#h)	(5 points) Based on your model, what values of group_size, homeowner, and married_couple will maximize the odds value Prob(A=1) / Prob(A = 0)?  
#What is that maximum odd value?
maximum = max(df_nb_predicted_prbabilities["1"]/df_nb_predicted_prbabilities["0"])


# In[76]:


all_combi[7]


# In[77]:


print("The maximum odd value is",maximum)
print("The combination for which this value occurs is",all_combi[7])

