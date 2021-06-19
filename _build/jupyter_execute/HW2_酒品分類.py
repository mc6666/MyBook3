#!/usr/bin/env python
# coding: utf-8

# # Wine 分類

# In[1]:


from sklearn.datasets import load_wine
ds = load_wine()


# In[2]:


print(ds.DESCR)


# In[3]:


import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()


# In[4]:


y = ds.target
y


# In[5]:


X.isnull().sum()


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test[0]


# In[8]:


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[9]:


knn.score(X_test, y_test)


# In[10]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)


# In[11]:


svc.score(X_test, y_test)


# In[12]:


from sklearn.metrics import confusion_matrix

y_pred=knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))


# In[ ]:




