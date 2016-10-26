
# coding: utf-8
import sys

traindata=sys.argv[1]
modelname=sys.argv[2]
# In[2]:

import pandas as pd
import numpy as np
import math
d = pd.read_csv(traindata, encoding='big5',header=None)


# In[3]:

trainData = d.values
trainData = trainData[:,1:]
n = trainData.shape[1]-1
m = trainData.shape[0]


# In[4]:

trainData1=np.matrix(trainData[np.where(trainData[:,n]==1),:n])
trainData0=np.matrix(trainData[np.where(trainData[:,n]==0),:n])
n1 = trainData1.shape[0]
n0 = trainData0.shape[0]


# In[5]:

from numpy import linalg
cov1=np.cov(trainData1.T)
cov0=np.cov(trainData0.T)
cov=np.matrix(cov0*(n0/m)+cov1*(n1/m))
mu = np.zeros(shape=(n,2))
for i in range(n):
    mu[i,0] = np.mean(trainData0[:,i])
    mu[i,1] = np.mean(trainData1[:,i])
mu = np.matrix(mu)


# In[6]:

pinv_cov = linalg.pinv(cov)


# In[7]:

w = np.matrix(mu[:,0]-mu[:,1]).T * pinv_cov


# In[8]:

b = (-mu[:,0].T * pinv_cov*mu[:,0] + mu[:,1].T * pinv_cov*mu[:,1])/2 + math.log(n0/n1)


# In[9]:

from model import a


# In[11]:

weight = a(w,b)


# In[13]:

import pickle
modelname = modelname+".pkl"
with open(modelname,'wb') as output:
    pickle.dump(weight,output)

