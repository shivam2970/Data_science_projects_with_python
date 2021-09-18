#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECCESARY  LIBRARIES

# In[1]:


import  pandas as  pd
import numpy  as np
import matplotlib.pyplot as plt


# In[2]:


pip install seaborn 


# In[3]:


import seaborn as  sns


# In[5]:


data=pd.read_csv('heart_failure_clinical_records_dataset.csv')
data.head()


# In[6]:


data.shape


# In[11]:


data.describe()


# In[7]:


data.isnull().values.any()


# # COORELATION MATRIX

# In[9]:


corrmat=data.corr()
top_corr_features   = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[12]:


data.corr()


# In[18]:


Outcome_True_data=len(data.loc[data['DEATH_EVENT']==1])
Outcome_False_data=len(data.loc[data['DEATH_EVENT']==0])


# In[19]:


Outcome_True_data, Outcome_False_data


# # IMPORTING TRAIN TEST SPLIT

# In[20]:


from sklearn.model_selection import train_test_split
features_columns=['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking']
predicted_class=['DEATH_EVENT']


# In[21]:


x=data[features_columns].values
y=data[predicted_class].values


# In[ ]:





# In[124]:


x_train,  x_test,  y_train,  y_test=train_test_split(x,  y, test_size=0.2, random_state=10)


# In[125]:


print('number of  rows missing age:{0}'.format(len(data.loc[data['age']==0])))


# In[126]:


print('train set', x_train.shape,  y_train.shape)
print('test set',  x_test.shape,  y_test.shape)


# # Using KNN ALGORITHM

# In[127]:


from sklearn.neighbors import KNeighborsClassifier


# In[128]:


k=4


# In[129]:


neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)


# In[130]:


neigh


# In[131]:


yhat=neigh.predict(x_test)
yhat[0:5]


# In[132]:


from sklearn  import metrics


# In[133]:


# Importing and fitting KNN classifier for k=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)


# # PREDICTION FOR  KNN

# In[134]:


# Predicting results using Test data set
pred = knn.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)


# # using RANDOM FOREST CLASSIFIER ALGORITHM

# In[135]:


from sklearn.ensemble import RandomForestClassifier


# In[148]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, stratify=y)


# In[149]:


forest = RandomForestClassifier()
forest.fit(x_train, y_train)


# # PREDICTION FOR RANDOM FOREST CLASSIFIER

# In[150]:


y_pred_test = forest.predict(x_test)


# In[151]:


accuracy_score(y_test, y_pred_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




