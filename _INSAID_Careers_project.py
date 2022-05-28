#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


#Reading the Data
df = pd.read_csv('Fraud.csv')
df.head()


# In[3]:


#finding out the null values
df.isnull().sum()


# In[4]:


#finding out correlation between the features
df.corr()


# In[5]:


df[df['isFraud']==1]


# In[6]:


df.info()


# In[7]:


#Data cleaning
df1=df.drop(['step','nameOrig','nameDest','isFlaggedFraud'],axis=1)


# In[8]:


df2=pd.get_dummies(df,columns=['type'])
df2.sample(4)


# In[9]:


x=df2.loc[:,['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']]
y=df2.loc[:,'isFraud']


# In[10]:


mx=MinMaxScaler()
mx.fit(x)
x=pd.DataFrame(mx.transform(x),columns=x.columns)
x


# In[11]:


# Splitting train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=1)


# In[12]:


df3=df2.sample(200)


# In[13]:


#finding the collinearity
sns.pairplot(df3)
plt.figure()
plt.show()


# In[14]:


#model is selected by checking every classification model accuracy 
#applying logistic regression first

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train , y_train)


# In[15]:


pred= lr.predict(x_test)
pred


# In[16]:


from sklearn import metrics
print ("Accuracy : ", metrics.accuracy_score(y_test, pred))


# In[17]:


from sklearn.metrics import confusion_matrix


# In[18]:


confusion_matrix(y_test, pred)


# In[ ]:




