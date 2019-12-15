#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
import sklearn.neural_network as nn
import sklearn.metrics as metrics 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


#Import Dataset
df = pd.read_csv('C:/Users/kinja/OneDrive/Desktop/All/1. SEM1/ML/Assignment5/SpiralWithCluster.csv')


# In[3]:


df.head()


# #Please answer the following questions based on your model.
# #a)	(5 points) What percent of the observations have SpectralCluster equals to 1? 

# In[43]:


count_all = df['SpectralCluster'].count()
count_one = df.SpectralCluster[df.SpectralCluster == 1].count()
percent_one = (count_one/count_all)*100
print("Percent of the observations have SpectralCluster equals to 1: ",percent_one)


# b)	(15 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification 
# rate.  You will use your answer in 
# (a) as the threshold for classifying an observation into SpectralCluster = 1. 
# Your search will be done over a grid that is formed by cross-combining the following attributes: 
#     (1) activation function: 
#         identity, logistic, relu, and tanh; 
#     (2) number of hidden layers: 1, 2, 3, 4, and 5; and 
#     (3) number of neurons: 
#         1 to 10 by 1.  List your optimal neural network for each activation function in a table.  
#         Your table will have four rows, one for each activation function.  
# Your table will have five columns: 
#     (1) activation function, 
#     (2) number of layers, 
#     (3) number of neurons per layer, 
#     (4) number of iterations performed, 
#     (5) the loss value, and 
#     (6) the misclassification rate.

# You are asked to use the Multi-Layer Perceptron (MLP) algorithm to classify SpectralCluster.  You will use the sklearn.neural_network.MLPClassifier function with the following specifications.
#     1.	Each hidden layer will have the same number of neurons 
#     2.	The initial learning rate is 0.1 --
#     3.	The maximum number of iterations is 5000 --
#     4.	The random seed is 20191108 --
#     5.	The solver for weight optimization is lbfgs --

# In[5]:


# Build Neural Network
def neural_network(function, layer, neurons):
    clf = nn.MLPClassifier(solver='lbfgs', learning_rate_init = 0.1, max_iter = 5000,activation = function, 
                  hidden_layer_sizes= (neurons,)*layer, random_state= 20191108) 
    predic = df[["x","y"]]
    target = df.SpectralCluster
    thisFit = clf.fit(predic, target) 
    y_pred = clf.predict(predic)
    
    labels = clf.classes_ 
    Loss = clf.loss_
    iterate = clf.n_iter_
    output_activation = clf.out_activation_ 
    RSquare = metrics.r2_score(target, y_pred) 
    accuracy = accuracy_score(target, y_pred)
    misclassify = 1 - accuracy 
    
    return Loss, RSquare, misclassify, iterate, labels, output_activation


# In[55]:


stats_ls = []

activation_function = ['identity', 'logistic', 'relu','tanh']
hidden_layer = 5
neurons_no = 10 

for function in activation_function: 
    for layer in range(1, hidden_layer+1):
        for neurons in range(1, neurons_no+1):
            Loss, RSquare, misclassify, iterate, labels, output_activation = neural_network(function, layer, neurons)
            stats_ls.append([function, layer, neurons,RSquare, iterate, Loss, misclassify])
stats_df = pd.DataFrame(stats_ls, columns = ['Activation function', 'Number of layers', 'Neurons per layer', 'RSquare',
                                            'Iterations performed','Loss value', 'Misclassification rate']) 


# In[48]:


stats_df.sort_values(by=['Loss value','Misclassification rate']).head()


# In[56]:


stats_df_relu = stats_df[stats_df["Activation function"] == "relu"]
stats_df_identity = stats_df[stats_df["Activation function"] == "identity"]
stats_df_logistic = stats_df[stats_df["Activation function"] == "logistic"]
stats_df_tanh = stats_df[stats_df["Activation function"] == "tanh"]

relu_frame = stats_df_relu[stats_df_relu["Loss value"] == stats_df_relu["Loss value"].min()]
identity_frame = stats_df_identity[stats_df_identity["Loss value"] == stats_df_identity["Loss value"].min()]
logistics_frame = stats_df_logistic[stats_df_logistic["Loss value"] == stats_df_logistic["Loss value"].min()]
tanh_frame = stats_df_tanh[stats_df_tanh["Loss value"] == stats_df_tanh["Loss value"].min()]

frames = [relu_frame, identity_frame, logistics_frame, tanh_frame]
result_table = pd.concat(frames)
result_table


# c)	(5 points) What is the activation function for the output layer?
# 
# Answer: Activation Functions for Output Layer for Interval type.
#         Customize your output activation function by assigning the function name to the out_activation_ of 
#         the Neural Network object.
#         out_activation_ : Name of the output activation function.

# In[39]:


print("Name of the output activation function: ",output_activation)


# d)	(5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?

# In[57]:


stats_df_relu = stats_df[stats_df["Activation function"] == "relu"]
stats_df_identity = stats_df[stats_df["Activation function"] == "identity"]
stats_df_logistic = stats_df[stats_df["Activation function"] == "logistic"]
stats_df_tanh = stats_df[stats_df["Activation function"] == "tanh"]

relu_frame = stats_df_relu[stats_df_relu["Loss value"] == stats_df_relu["Loss value"].min()]
identity_frame = stats_df_identity[stats_df_identity["Loss value"] == stats_df_identity["Loss value"].min()]
logistics_frame = stats_df_logistic[stats_df_logistic["Loss value"] == stats_df_logistic["Loss value"].min()]
tanh_frame = stats_df_tanh[stats_df_tanh["Loss value"] == stats_df_tanh["Loss value"].min()]

frames = [relu_frame, identity_frame, logistics_frame, tanh_frame]
result_table = pd.concat(frames)
result_table


# In[11]:


optimal_clf = nn.MLPClassifier(solver='lbfgs', learning_rate_init = 0.1, max_iter = 5000,activation = 'relu', 
                  hidden_layer_sizes= (8,)*4, random_state = 20191108)
predic = df[["x","y"]]
target = df.SpectralCluster
thisFit = optimal_clf.fit(predic, target) 
optimal_y_pred = optimal_clf.predict(predic)
optimal_clf_pred_prob = optimal_clf.predict_proba(predic)
df['NLPpredictions'] = optimal_y_pred
pred_proba = pd.DataFrame(data=optimal_clf_pred_prob,columns = ["clas0","class1"])


# e)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')
colors = ['red','blue']
for i in range(2):
    Data = df[df['NLPpredictions']==i]
    plt.scatter(Data.x,Data.y,c = colors[i],label=i)
    #plt.legend()
plt.title("Scatterplot according to Cluster Values Predicted by optimal neural network")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()


# f)	(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places

# In[38]:


#From Q1.a, we get the threshold to be 0.5:
#Hence:
pred_prob_class1 = pred_proba[pred_proba["class1"] > 0.5]["class1"]
pred_prob_class1 = pred_prob_class1.to_list()
print("Count of predicted probability Prob(SpectralCluster = 1): ",len(pred_prob_class1))
print("Mean of predicted probability Prob(SpectralCluster = 1): ",round(np.mean(pred_prob_class1),10))
print("Standard Deviation of predicted probability Prob(SpectralCluster = 1): ",round(np.std(pred_prob_class1),10))


# Q2

# You are asked to use the Support Vector Machine (SVM) algorithm to classify SpectralCluster.  You will use the sklearn.svm.SVC function with the following specifications.
# 1.	The linear kernel
# 2.	The decision function shape is One Over Rest (OVR)
# 3.	No limit on the number of iterations
# 4.	The random seed is 20191108
# 

# b) (5 points) What is the misclassification rate?

# In[16]:


from sklearn.svm import SVC

predic = df[["x","y"]]
target = df.SpectralCluster
svm_clf = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
svm_clf.fit(predic,target)
svm_pred = svm_clf.predict(predic)
svm_accuracy = accuracy_score(target, svm_pred)
svm_missclassification = 1- svm_accuracy
print("The miscalssification rate is: ", svm_missclassification)


# c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[17]:


df["SVM_pred"] = svm_pred


# In[18]:


coeff = svm_clf.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm_clf.intercept_[0]) / coeff[1]

# plot the line, the points, and the nearest vectors to the plane
carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(xx, yy, 'k--')


for i in range(2):
    Data = df[df["SVM_pred"]==i]
    plt.scatter(Data.x,Data.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of Cluster Values")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# a) (5 points) What is the equation of the separating hyperplane? Please state the coefficients up to seven decimal places.

# In[19]:


#coefficients rounded to seven decimal places
print ('Equation of the seperating hyperplane is')
print (round(svm_clf.intercept_[0],7), " + (", round(coeff[0],7), ") X +(" ,round(coeff[1],7),") Y = 0")


# (d) (10 points) Please express the data as polar coordinates. Please plot the theta-coordinate against the radius-coordinate in a scatterplot. Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue). To obtain the full credits, you should properly label the axes, the legend, and the chart title. Also, grid lines should be added to the axes.

# In[40]:


#Function for Theta transformation normalization:
def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

trainData = pd.DataFrame(columns = ["radius","theta"])
trainData['radius'] = np.sqrt(df['x']**2 + df['y']**2)
trainData['theta'] = (np.arctan2(df['y'], df['x'])).apply(customArcTan)
trainData['class']=df["SpectralCluster"]
trainData.head()


# In[41]:


colour = ['red','blue']
for i in range(2):
    Data = trainData[trainData["class"]==i]
    plt.scatter(Data.radius,Data.theta,label = (i),c = colour[i])
    
plt.title("Scatterplot of Co-ordinates")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()
plt.grid()


# (e) (10 points) You should expect to see three distinct strips of points and a lone point. Since the SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of the chart in (d), values 1, 2,and 3 for the next three strips of points.
# 
# Please plot the theta-coordinate against the radius-coordinate in a scatterplot. Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). To obtain the full credits, you should properly label the axes, the legend, and the chart title. Also, grid lines should be added to the axes.

# In[42]:


x = trainData["radius"]
y = trainData['theta'].apply(customArcTan)
svm_df = pd.DataFrame(columns = ['Radius','Theta'])
svm_df['Radius'] = x
svm_df['Theta'] = y

group = []

for i in range(len(x)):
    if x[i] < 1.5 and y[i]>6:
        group.append(0)
        
    elif x[i] < 2.5 and y[i]>3 :
        group.append(1)
    
    elif 2.75 > x[i]>2.5 and y[i]>5:
        group.append(1)
        
    elif 2.5<x[i]<3 and 2<y[i]<4:
        group.append(2)   
    
    elif x[i]> 2.5 and y[i]<3.1:
        group.append(3)
        
    elif x[i] < 4:
        group.append(2)
        

svm_df['Group'] = group
colors = ['red','blue','green','black']
for i in range(4):
    Data = svm_df[svm_df.Group == i]
    plt.scatter(Data.Radius,Data.Theta,c = colors[i],label=i)
plt.grid()
plt.title("Scatterplot with four Groups")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()


# (g) (5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot. Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title. Also, grid lines should be added to the axes.

# In[53]:


#SVM to classify class 0 and class 1
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 0]
x = x.append(svm_df[svm_df['Group'] == 1])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

coeff = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(1, 2)
yy = a * xx - (svm_1.intercept_[0])/coeff[1] 

print ('Equation of the hypercurve for SVM 0 is')
print (svm_1.intercept_[0], " + (", coeff[0], ") X +(" ,coeff[1],") Y = 0")

h0_xx = xx * np.cos(yy[:])
h0_yy = xx * np.sin(yy[:])

carray=['red','blue','green','black']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to classify class 1 and class 2
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 1]
x = x.append(svm_df[svm_df['Group'] == 2])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

w = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(1, 4)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('Equation of the hypercurve for SVM 1 is')
print (svm_1.intercept_[0], " + (", coeff[0], ") X +(" ,coeff[1],") Y = 0")

h1_xx = xx * np.cos(yy[:])
h1_yy = xx * np.sin(yy[:])

#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to. classify class 2 and class 3
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_df[svm_df['Group'] == 2]
x = x.append(svm_df[svm_df['Group'] == 3])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Group)

coeff = svm_1.coef_[0]
a = -coeff[0] / coeff[1]
xx = np.linspace(2, 4.5)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('Equation of the hypercurve for SVM 2 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h2_xx = xx * np.cos(yy[:])
h2_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

for i in range(4):
    Data = svm_df[svm_df.Group == i]
    plt.scatter(Data.Radius,Data.Theta,c = carray[i],label=i)
plt.xlabel("Radius")
plt.ylabel("Theta Co-Ordinate")
plt.title("Theta-coordinate against the Radius-coordinate in a scatterplot seperated by 3 hyperplanes")
plt.legend()


# (h) (10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot. Please color-code the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title. Also, grid lines should be added to the axes.
# 
# Based on your graph, which hypercurve do you think is not needed?

# In[54]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    Data = df[df["SpectralCluster"]==i]
    plt.scatter(Data.x,Data.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of the cartesian co-ordinates with hypercurves")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# In[ ]:




