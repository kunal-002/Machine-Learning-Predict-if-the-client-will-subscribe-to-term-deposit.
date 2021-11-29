#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import seaborn as sn


# In[4]:


train = pd.read_csv("train.csv")


# In[85]:


test = pd.read_csv("test.csv")


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.describe()


# In[13]:


train.shape


# In[14]:


test.shape


# In[15]:


train.dtypes


# In[9]:


# Univariate analysis
train["age"].hist()


# In[10]:


train.boxplot("age")


# In[24]:


train["subscribed"].value_counts()


# In[25]:


train["subscribed"].value_counts().plot.bar()


# In[26]:


train["subscribed"].value_counts(normalize=True)


# In[20]:


train[["subscribed","loan"]].value_counts().plot.bar()


# In[21]:


train["poutcome"].value_counts()


# In[22]:


train[["poutcome","subscribed"]].value_counts()


# In[23]:


train[["poutcome","subscribed"]].value_counts().plot.bar()


# In[28]:


train["job"].value_counts()


# In[29]:


train["job"].value_counts().plot.bar()


# In[34]:


train["default"].value_counts()


# In[35]:


train["default"].value_counts().plot.bar()


# In[36]:


#Bivariate analysis
print(pd.crosstab(train['job'],train['subscribed']))


# In[42]:


job=pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# In[43]:


print(pd.crosstab(train['default'],train['subscribed']))

default=pd.crosstab(train['default'],train['subscribed'])
default.div(default.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


# In[44]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[49]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# In[52]:


train.isnull().sum()


# In[53]:


#classifiaction problem so logistic regression and decsion tree 


# In[54]:


target = train["subscribed"]


# In[55]:


train = train.drop("subscribed",1)


# In[56]:


# applying dummies on the train dataset
train = pd.get_dummies(train)


# In[57]:


train.head()


# In[58]:


train.shape


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


# In[61]:


# Builiding logistic regression model


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


lreg = LogisticRegression()


# In[65]:


lreg.fit(X_train,y_train)


# In[66]:


prediction = lreg.predict(X_val)


# In[68]:


# Calculating accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_val,prediction)


# In[86]:


# test data
test = pd.get_dummies(test)
predict_test = lreg.predict(test)


# In[87]:


predict_test = lreg.predict(test)


# In[88]:


predict_test


# In[89]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[122]:


clf = DecisionTreeClassifier(max_depth=5,random_state=0)


# In[123]:


clf.fit(X_train,y_train)


# In[124]:


pred = clf.predict(X_val)


# In[125]:


accuracy_score(y_val,pred)


# In[127]:


test_prediction = clf.predict(test)


# In[128]:


submission = pd.DataFrame()


# In[129]:


submission["ID"] = test["ID"]
submission["Subscribed"] = test_prediction


# In[130]:


submission["Subscribed"].replace(0,"no",inplace=True)
submission["Subscribed"].replace(1,"Yes",inplace=True)


# In[131]:


submission.to_csv("submission.csv",header=True,index=False)


# In[133]:


test["subscribed"] = test_prediction


# In[134]:


test.columns


# In[ ]:




