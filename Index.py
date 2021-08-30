#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas  as pd
import  matplotlib.pyplot as  plt


# In[6]:


dataset = pd.read_csv("diabetes.csv")
dataset.head()


# In[7]:


dataset.shape


# In[8]:


dataset.isnull().values.any()


# In[9]:


pip install seaborn


# In[10]:


import seaborn as sns


# In[16]:


corrmat=dataset.corr()
top_corr_features   = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[18]:


dataset.corr()


# # Changing   the   Outcome data column to boolean

# In[19]:


Outcome_map={True:1, False:0}


# In[22]:


dataset.head()


# In[25]:


Outcome_True_data=len(dataset.loc[dataset['Outcome']==True])
Outcome_False_data=len(dataset.loc[dataset['Outcome']==False])


# In[26]:


Outcome_True_data, Outcome_False_data


# # Training the model

# In[33]:


from sklearn.model_selection import train_test_split
features_columns=['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',  'BMI', 'DiabetesPedigreeFunction', 'Age' ]
predicted_class=['Outcome']


# In[34]:


x=dataset[features_columns].values
y=dataset[predicted_class].values


# In[35]:


x_train,  x_test,  y_train,  y_test=train_test_split(x,  y, test_size=0.2, random_state=10)


# # Check missing(0) Values

# In[39]:


print('number of  rows missing Glucose:{0}'.format(len(dataset.loc[dataset['Glucose']==0])))
print('number of  rows missing BloodPressure:{0}'.format(len(dataset.loc[dataset['BloodPressure']==0])))
print('number of  rows missing SkinThickness:{0}'.format(len(dataset.loc[dataset['SkinThickness']==0])))
print('number of  rows missing BMI:{0}'.format(len(dataset.loc[dataset['BMI']==0])))
print('number of  rows missing DiabetesPedigreeFunction:{0}'.format(len(dataset.loc[dataset['DiabetesPedigreeFunction']==0])))
print('number of  rows missing Insulin:{0}'.format(len(dataset.loc[dataset['Insulin']==0])))
print('number of  rows missing Age:{0}'.format(len(dataset.loc[dataset['Age']==0])))




# In[57]:




mean_Glucose=dataset['Glucose'].mean()
dataset=dataset.replace({'Glucose': {0: mean_Glucose}}) 


# In[62]:


mean_SkinThickness=dataset['SkinThickness'].mean()
dataset=dataset.replace({'SkinThickness': {0: mean_SkinThickness}}) 


# In[60]:


mean_BloodPressure=dataset['BloodPressure'].mean()
dataset=dataset.replace({'BloodPressure': {0: mean_BloodPressure}}) 


# In[63]:


mean_BMI=dataset['BMI'].mean()
dataset=dataset.replace({'BMI': {0: mean_BMI}}) 


# In[64]:




mean_Insulin=dataset['Insulin'].mean()
dataset=dataset.replace({'Insulin': {0: mean_Insulin}}) 


# In[65]:


print('number of  rows missing Glucose:{0}'.format(len(dataset.loc[dataset['Glucose']==0])))
print('number of  rows missing BloodPressure:{0}'.format(len(dataset.loc[dataset['BloodPressure']==0])))
print('number of  rows missing SkinThickness:{0}'.format(len(dataset.loc[dataset['SkinThickness']==0])))
print('number of  rows missing BMI:{0}'.format(len(dataset.loc[dataset['BMI']==0])))
print('number of  rows missing DiabetesPedigreeFunction:{0}'.format(len(dataset.loc[dataset['DiabetesPedigreeFunction']==0])))
print('number of  rows missing Insulin:{0}'.format(len(dataset.loc[dataset['Insulin']==0])))
print('number of  rows missing Age:{0}'.format(len(dataset.loc[dataset['Age']==0])))


# In[66]:


from sklearn.ensemble import RandomForestClassifier


# In[69]:


random_forest_model=RandomForestClassifier(random_state=10)
random_forest_model.fit(x_train,y_train.ravel())


# In[83]:


predict_data=random_forest_model.predict(x_test)
from  sklearn  import  metrics
print("accuracy"metrics.accuracy_score(y_train))


# In[75]:





# In[ ]:




