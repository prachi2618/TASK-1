#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[23]:


train_data=pd.read_csv('C:\\Users\\PRACHI\\Downloads\\digit-recognizer\\train.csv')
test_data=pd.read_csv('C:\\Users\\PRACHI\\Downloads\\digit-recognizer\\test.csv')
submission_data=pd.read_csv('C:\\Users\\PRACHI\\Downloads\\digit-recognizer\\sample_submission.csv')
data.shape


# In[11]:


data.head(3)


# In[26]:


X=train_data.iloc[:,1:].values
X.shape


# In[27]:


Y=train_data.iloc[:,0].values
Y.shape


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)


# In[29]:


X_train.shape


# In[31]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()


# In[32]:


classifier.fit(X_train,Y_train)


# In[34]:


Y_test[100]


# In[36]:


plt.imshow(X_test[100].reshape(28,28))
classifier.predict(X_test[100].reshape(1,784))


# In[ ]:




