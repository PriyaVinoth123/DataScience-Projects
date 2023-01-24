#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


#Data_collection and reading the first five rows of the dataset
df = pd.read_csv("cardio_dataset.csv")
df.head()


# In[4]:


#print last five rows of the dataset
df.tail()


# In[9]:


#few info about the dataset
df.info()


# In[8]:


#no of rows and columns in the dataset
df.shape


# In[10]:


#checking for missing values
df.isnull().sum(axis = 0)


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['age'])
plt.show()

#No Outliers observed in 'age'
sns.boxplot(x=df['sex'])
plt.show()

#No outliers observed in sex data
sns.boxplot(x=df['cp'])
plt.show()

#No outliers in 'cp'
sns.boxplot(x=df['trestbps'])
plt.show()

#Some outliers are observed in 'trtbps'. 
sns.boxplot(x=df['chol'])
plt.show()

#Some outliers are observed in 'chol'.
sns.boxplot(x=df['fbs'])
plt.show()

sns.boxplot(x=df['restecg'])
plt.show()

sns.boxplot(x=df['thalach'])
plt.show()
#Outliers present in thalachh


# In[12]:


#statistical measures of the data
df.describe()


# In[13]:


#checking the distribution of target variable
df['target'].value_counts


# In[15]:


#1 -> defective heart, 0 -> healthy heart
#splitting the features and target

X = df.drop(columns = 'target', axis =1)
Y = df['target']
print(X)


# In[16]:


print(Y)


# In[18]:


#Splitting the data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_test.shape,X_train.shape)


# In[20]:


#Logistic Regression
model = LogisticRegression()
#Training a model with training data
model.fit(X_train,Y_train)


# In[21]:


#Model Evaluation
#Accuracy score
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data:',training_data_accuracy)


# In[22]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data:',training_data_accuracy)


# In[28]:


#Building a predictive system
import warnings
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)

#changing the input data into a numpy array
input_data_as_numpy_array = np.array(input_data)

#reshaping the numpy array as we predict it
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)

if(prediction[0] == 0):
    print('The person does not have heart disease')
else:
    print('The person has heart disease')


# In[ ]:




