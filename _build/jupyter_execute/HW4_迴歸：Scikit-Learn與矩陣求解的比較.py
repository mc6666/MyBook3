#!/usr/bin/env python
# coding: utf-8

# # 迴歸：Scikit-Learn 與矩陣求解的比較

# In[1]:


from sklearn import datasets
ds= datasets.load_boston()


# In[2]:


print(ds.DESCR)


# In[3]:


import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target
X.head(10)


# In[4]:


y


# In[5]:


X.info()


# In[6]:


import numpy as np
X.AGE.astype(np.float32)


# In[7]:


X.isnull().sum()


# In[8]:


X.isnull().sum().sum()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape, X_test.shape


# In[11]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[12]:


lr.coef_


# In[13]:


lr.intercept_


# In[14]:


lr.score(X_test, y_test)


# In[15]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, lr.predict(X_test))


# In[16]:


mean_squared_error(y_test, lr.predict(X_test)) ** .5


# In[17]:


from sklearn.metrics import r2_score
r2_score(y_test, lr.predict(X_test))


# ## 二次迴歸

# In[18]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X2 = poly.fit_transform(X)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2)


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape, X_test.shape


# In[21]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[22]:


lr.score(X_test, y_test)


# In[23]:


len(lr.coef_)


# ## 簡單運算

# In[24]:


import numpy as np

A = np.array([[2,4],
              [6,2]])

B = np.array([[18],
              [34]])

C = np.linalg.solve(A, B)

print(C)


# In[25]:


np.linalg.inv(A) @ B


# In[26]:


np.linalg.inv(A.T @ A) @ A.T @ B


# In[27]:


from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")

X=A
y=B

X1=X[:, 0].reshape(X.shape[0])
X2=X[:, 1].reshape(X.shape[0])
ax.scatter3D(X1, X2, y, cmap='hsv', marker= 'o', s = [160,160])

X1=np.linspace(2,8,50)
X2=np.linspace(2,8,50)
x_surf, y_surf = np.meshgrid(X1, X2)
z_surf= x_surf *5 +  y_surf * 2 
from matplotlib import cm
ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot)    # plot a 3d surface plot
plt.show()


# ## Boston by matrix

# In[28]:


from sklearn import datasets
ds= datasets.load_boston()


# In[29]:


import pandas as pd

X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = ds.target
X.head(10)


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[31]:


b=np.ones((X_train.shape[0], 1))
b.shape


# In[32]:


X_train=np.hstack((X_train, b))


# In[33]:


# np.linalg.inv(A.T @ A) @ A.T @ B
W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
W


# In[34]:


X_test.shape, W.shape, y_test.shape


# In[35]:


b=np.ones((X_test.shape[0], 1))
b.shape


# In[36]:


X_test=np.hstack((X_test, b))


# In[37]:


SSE = ((X_test @ W - y_test ) ** 2).sum() 
MSE = SSE / y_test.shape[0]
MSE


# In[38]:


RMSE = MSE ** (1/2)
RMSE


# ## R^2 公式
# https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

# In[39]:


y_mean = y_test.ravel().mean()
SST = ((y_test - y_mean) ** 2).sum()
R2 = 1 - (SSE / SST)
R2


# In[ ]:




