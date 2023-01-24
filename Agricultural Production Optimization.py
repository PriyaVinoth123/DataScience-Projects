#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from ipywidgets import interact


# In[23]:


agridata = pd.read_csv('AgricultureDS.csv')


# In[24]:


print("Shape of the dataset =",agridata.shape)


# In[25]:


agridata.head()


# In[26]:


agridata.isnull().sum()


# In[27]:


agridata['label'].value_counts()


# In[28]:


print("Average ratio of Nitrogen in the soil:{0:.2f}".format(agridata['N'].mean()))
print("Average ratio of Potassium in the soil:{0:.2f}".format(agridata['K'].mean()))
print("Average ratio of Phosphorus in the soil:{0:.2f}".format(agridata['P'].mean()))
print("Average temperature in celsius :{0:.2f}".format(agridata['temperature'].mean()))
print("Average humidity in % :{0:.2f}".format(agridata['humidity'].mean()))
print("Average ph of the soil :{0:.2f}".format(agridata['ph'].mean()))
print("Average rainfall in mm :{0:.2f}".format(agridata['rainfall'].mean()))


# In[29]:


@interact
def summary(crops = list(agridata['label'].value_counts().index)):
    x = agridata[agridata['label'] == crops]
    print("--------------------------------------------------------")
    print("Statistics for Nitrogen:")
    print("Minimum Nitrogen required:",x['N'].min())
    print("Average Nitrogen required:",x['N'].mean())
    print("Maximum Nitrogen required:",x['N'].max())
    print("--------------------------------------------------------")
    print("Statistics for Phosphorus:")
    print("Minimum Phosphoprus required:",x['P'].min())
    print("Average Phosphorus required:",x['P'].mean())
    print("Maximum Phosphorus required:",x['P'].max())
    print("--------------------------------------------------------")
    print("Statistics for Potassium:")
    print("Minimum Potassium required:",x['K'].min())
    print("Average Potassium required:",x['K'].mean())
    print("Maximum Potassium required:",x['K'].max())
    print("--------------------------------------------------------")
    print("Statistics for PH")
    print("Minimum PH required:{0:.2f}".format(x['ph'].min()))
    print("Average PH required:{0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required:{0:.2f}".format(x['ph'].max()))
    print("--------------------------------------------------------")
    print("Stat")
    print("Minimum Rainfall required:{0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required:{0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required:{0:.2f}".format(x['rainfall'].max()))
    print("--------------------------------------------------------")


# In[30]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average values for",conditions,"is {0:.2f}".format(agridata[conditions].mean()))
    print("--------------------------------------------------------")
    print("Rice :{0:.2f}".format(agridata[(agridata['label'] == 'rice')][conditions].mean()))
    print("Blackgrams :{0:.2f}".format(agridata[(agridata['label'] == 'blackgram')][conditions].mean()))
    print("Maize :{0:.2f}".format(agridata[(agridata['label'] == 'maize')][conditions].mean()))
    print("Chick peas :{0:.2f}".format(agridata[(agridata['label'] == 'chickpea')][conditions].mean()))
    print("mothbeans :{0:.2f}".format(agridata[(agridata['label'] == 'mothbeans')][conditions].mean()))
    print("kidneybeans :{0:.2f}".format(agridata[(agridata['label'] == 'kidneybeans')][conditions].mean()))
    print("pigeonpeas:{0:.2f}".format(agridata[(agridata['label'] == 'pigeonpeas')][conditions].mean()))
    print("mungbean :{0:.2f}".format(agridata[(agridata['label'] == 'mungbean')][conditions].mean()))
    print("lentil :{0:.2f}".format(agridata[(agridata['label'] == 'lentil')][conditions].mean()))
    print("pomegranate :{0:.2f}".format(agridata[(agridata['label'] == 'pomegranate')][conditions].mean()))
    print("banana :{0:.2f}".format(agridata[(agridata['label'] == 'banana')][conditions].mean()))
    print("grapes :{0:.2f}".format(agridata[(agridata['label'] == 'grapes')][conditions].mean()))
    print("watermelon :{0:.2f}".format(agridata[(agridata['label'] == 'watermelon')][conditions].mean()))
    print("muskmelon :{0:.2f}".format(agridata[(agridata['label'] == 'muskmelon')][conditions].mean()))
    print("apple :{0:.2f}".format(agridata[(agridata['label'] == 'apple')][conditions].mean()))
    print("orange :{0:.2f}".format(agridata[(agridata['label'] == 'orange')][conditions].mean()))
    print("papaya :{0:.2f}".format(agridata[(agridata['label'] == 'papaya')][conditions].mean()))
    print("coconut :{0:.2f}".format(agridata[(agridata['label'] == 'coconut')][conditions].mean()))
    print("jute :{0:.2f}".format(agridata[(agridata['label'] == 'jute')][conditions].mean()))
    print("cotton :{0:.2f}".format(agridata[(agridata['label'] == 'cotton')][conditions].mean()))
    print("coffee :{0:.2f}".format(agridata[(agridata['label'] == 'coffee')][conditions].mean()))
    print("--------------------------------------------------------")
    
    
    


# In[31]:


@interact
def compare(conditions = ['N','P','K','temperature','rainfall','humidity']):
    print("Crops that require greater than average",conditions,'\n')
    print(agridata[agridata[conditions]>agridata[conditions].mean()]['label'].unique())
    print("--------------------------------------------------------")
    print("Crops that require greater than average",conditions,'\n')
    print(agridata[agridata[conditions]<=agridata[conditions].mean()]['label'].unique())


# In[42]:


#distribution
plt.rcParams['figure.figsize'] = (15,7)

plt.subplot(2,4,1)
sns.distplot(agridata['N'],color = 'lightgrey')
plt.xlabel('Ratio of Nitrogen',fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(agridata['P'],color = 'skyblue')
plt.xlabel('Ratio of Phosphorus',fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(agridata['K'],color = 'darkblue')
plt.xlabel('Ratio of Potassium',fontsize = 12)
plt.grid()


plt.subplot(2,4,4)
sns.distplot(agridata['rainfall'],color = 'green')
plt.xlabel('Ratio of Rainfall',fontsize = 12)
plt.grid()


plt.subplot(2,4,5)
sns.distplot(agridata['ph'],color = 'brown')
plt.xlabel('Ratio of PH',fontsize = 12)
plt.grid()


plt.subplot(2,4,6)
sns.distplot(agridata['humidity'],color = 'yellow')
plt.xlabel('Ratio of Humidity',fontsize = 12)
plt.grid()


# In[43]:


#Patterns
print("Some interesting patterns in the Dataset")
print("----------------------------------------")
print("Crops which require high ratio of Nitrogen content in the soil:",agridata[agridata['N'] > 120]['label'].unique())
print("Crops which require high ratio of Phosphorous content in the soil:",agridata[agridata['P'] > 100]['label'].unique())
print("Crops which require high ratio of Potassium content in the soil:",agridata[agridata['K'] > 200]['label'].unique())
print("Crops which require high Rainfall:",agridata[agridata['rainfall'] > 200]['label'].unique())
print("Crops which require low Humidity:",agridata[agridata['humidity'] < 40]['label'].unique())
print("Crops which require high PH:",agridata[agridata['ph'] > 9]['label'].unique())
print("Crops which require high Temperature:",agridata[agridata['temperature'] > 40]['label'].unique())


# In[45]:


#Clustering
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

x = agridata.loc[:,['N','P','K','temperature','ph','humidity','rainfall']].values

print(x.shape)

x_data = pd.DataFrame(x)
x_data.head()


# In[46]:


#Cluster Determination
plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init  = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method',fontsize = 20)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show()


# In[49]:


#clustering analysis
km = KMeans(n_clusters = i, init  = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#lets find out the results
a = agridata['label']
y_means = pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns = {0:'cluster'})

#cluster for each crops
print("crops in first cluster:",z[z['cluster'] == 0]['label'].unique())
print('----------------------------------------------------------------')
print("crops in second cluster:",z[z['cluster'] == 1]['label'].unique())
print('----------------------------------------------------------------')
print("crops in third cluster:",z[z['cluster'] == 2]['label'].unique())
print('----------------------------------------------------------------')
print("crops in fourth cluster:",z[z['cluster'] == 3]['label'].unique())
print('----------------------------------------------------------------')


# In[51]:


# splitting the data for predictive modelling

y = agridata['label']
x = agridata.drop(['label'], axis = 1)

print("Shape of X:",x.shape)
print("Shape of Y:",y.shape)


# In[52]:


#Training and Testing

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

print("The Shape of X train:",x_train.shape)
print("The Shape of X test:",x_test.shape)
print("The Shape of Y train:",y_train.shape)
print("The Shape of Y test:",y_test.shape)


# In[59]:


#Predictive model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[60]:


#Accuracy of the model
from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred)
print(cr)


# In[61]:


agridata.head()


# In[62]:


prediction = model.predict((np.array([[90,
                                       40,
                                       40,
                                       20,
                                       80,
                                        7,
                                      200]])))
print("The suggested crop for given climatic condition is:",prediction)


# In[63]:


#Check for oranges
agridata[agridata['label'] == 'orange'].head()


# In[64]:


prediction = model.predict((np.array([[20,
                                       30,
                                       10,
                                       15,
                                       90,
                                        7.5,
                                      100]])))

print("The suggested crop for given climatic condition is:",prediction)


# In[ ]:




