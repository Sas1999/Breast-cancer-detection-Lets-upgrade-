#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df['Unnamed: 32']


# In[9]:


df = df.drop("Unnamed: 32", axis=1)


# In[10]:


df.head()


# In[11]:


df.columns


# In[12]:


df.drop('id', axis=1, inplace=True)
# df = df.drop('id', axis=1)


# In[13]:


df.columns


# In[14]:


type(df.columns)


# In[15]:


a=list(df.columns)
print(a)


# In[16]:


features_mean = a[1:11]

features_se = a[11:21]

features_worst = a[21:]


# In[17]:


print(features_mean)


# In[18]:


print(features_se)


# In[19]:


print(features_worst)


# In[20]:


df.head(2)


# In[21]:


df['diagnosis'].unique()
# M= Malignant, B= Benign


# In[24]:


sns.countplot(df['diagnosis'], label="Count")


# In[25]:


df['diagnosis'].value_counts()


# In[26]:


df.shape


# In[27]:


df.describe()


# In[28]:


len(df.columns)


# In[29]:


# Correlation Plot
corr = df.corr()
corr


# In[30]:


corr.shape


# In[31]:


plt.figure(figsize=(8,8))
sns.heatmap(corr);


# In[32]:


df.head()


# In[33]:


df['diagnosis']= df['diagnosis'].map({'M':1, 'B':0})


# In[34]:


df.head()


# In[35]:


df['diagnosis'].unique()


# In[36]:


X = df.drop('diagnosis', axis=1)
X.head()


# In[37]:


y = df['diagnosis']
y.head()


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[39]:


df.shape


# In[40]:


X_train.shape


# In[41]:


X_test.shape


# In[42]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


X_train.head(1)


# In[45]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[46]:


X_train


# In[47]:


#Logistic Regression


# In[48]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[49]:


y_pred = lr.predict(X_test)


# In[50]:


y_pred


# In[51]:


y_test


# In[52]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[53]:


lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[54]:


results = pd.DataFrame()
results


# In[55]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[56]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[57]:


y_pred = svc.predict(X_test)
y_pred


# In[58]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[59]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[60]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[ ]:





# In[ ]:




