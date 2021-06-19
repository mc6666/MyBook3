#!/usr/bin/env python
# coding: utf-8

# # 房價迴歸預測

# In[1]:


from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


ds = datasets.load_boston()


# In[3]:


print(ds.DESCR)


# In[4]:


import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()


# In[5]:


y = ds.target
y


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[7]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[8]:


scaler = preprocessing.StandardScaler()


# In[9]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


from sklearn.linear_model import LinearRegression 
clf = LinearRegression()


# In[11]:


clf.fit(X_train, y_train)


# In[12]:


clf.coef_


# In[13]:


clf.intercept_


# In[14]:


import numpy as np
np.argsort(abs(clf.coef_))


# In[15]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_squared_error(y_test, clf.predict(X_test))


# In[16]:


mean_absolute_error(y_test, clf.predict(X_test))


# In[17]:


r2_score(y_test, clf.predict(X_test))


# In[ ]:




