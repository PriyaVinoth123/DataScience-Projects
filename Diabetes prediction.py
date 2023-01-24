#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# In[5]:


# Data processing
diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()


# In[7]:


# no of rows and columns
diabetes_data.shape


# In[8]:


#statistical data from the dataset
diabetes_data.describe()


# In[10]:


diabetes_data['Outcome'].value_counts()


# In[12]:


diabetes_data.groupby('Outcome').mean()


# In[15]:


#dropping outcome
X = diabetes_data.drop(columns = 'Outcome', axis =1)
Y = diabetes_data['Outcome']
print(X)
print(Y)


# In[ ]:





# In[18]:


# splitting the data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, stratify=Y,random_state = 2)
print(X.shape,X_train.shape,X_test.shape)


# In[19]:


# Training the model
classifier = svm.SVC(kernel = 'linear')


# In[20]:


#Training SVM
classifier.fit(X_train, Y_train)


# In[22]:


#accuracy score training data
X_train_prediction  = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score:',training_data_accuracy)


# In[25]:


#accuracy score test data
X_test_prediction  = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score:',test_data_accuracy)


# In[ ]:




