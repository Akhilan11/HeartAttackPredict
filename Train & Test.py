#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
from sklearn.model_selection import train_test_split

# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[3]:


dataset = read_csv("heart.csv")


# In[4]:


print(dataset)


# In[5]:


array = dataset.values
X = array[:,0:13]
y = array[:,13]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=1)
# model = SVC(gamma='auto')
model = LogisticRegression(solver='liblinear', multi_class='ovr')


# In[6]:


model.fit(X_train, Y_train)


# In[7]:


import pickle
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[8]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_validation, Y_validation)
print(result)
#validation accuracy


# In[9]:


value = [[56,0,1,140,294,0,0,153,0,1.3,1,0,2]]
predictions = model.predict(value)
print(predictions[0])


# In[ ]:




