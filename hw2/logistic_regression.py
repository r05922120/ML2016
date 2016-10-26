
# coding: utf-8

# In[ ]:

import sys
trainingData = sys.argv[1]
modelname = sys.argv[2]


# In[3]:

import pandas as pd
import numpy as np
d = pd.read_csv(trainingData, encoding='big5',header=None)


# In[36]:

trainData = d.values
trainData = trainData[:,1:]


# In[5]:

X = np.matrix(trainData[:,:57])
y = np.matrix(trainData[:,57]).T
X = np.hstack([np.ones(shape=(X.shape[0],1)),X])


# In[6]:

def sigmod(z):
    z=np.matrix(z)
    return 1/(1+np.exp(-z))

def crossentropy(X,y,theta,londa):
    temp = sigmod(X*theta)
    y=np.matrix(y)
    m=y.shape[0]
    X=np.matrix(X)
    tempy = np.ones(shape=(y.shape))-y
    tempx = np.ones(shape=(temp.shape))-temp
    reg = londa*(np.sum(np.square(theta[1:,:])))/2
    return -(((y.T*np.log(temp))+(tempy).T*np.log(tempx))+reg)/m
    


# In[ ]:

def accuracy(predict,y):
    m=y.shape[0]
    return(1-(np.sum(np.abs(predict-y))/m))*100


# In[ ]:

def predict(X,theta):
    predict = np.matrix(np.zeros(shape=(X.shape[0],1)))
    count=0
    for i in np.array(sigmod(X*theta)):
        if i >=0.5:
            predict[count,0]=1
        else:
            predict[count,0]=0
        count+=1
    return predict


# In[9]:

def gradientdescent(X, y, theta, alpha, num_iters, londa):
    import numpy as np
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = X.shape[0]
    acc_history = np.zeros(num_iters)
    adagrad = np.zeros(theta.shape)
    for i in range(num_iters):
        temp = X.T * (sigmod(X * theta) - y)
        temp[1:] += londa * theta[1:]
        temp = (alpha/m) * temp
        adagrad += np.square(temp)
        temp = temp/np.sqrt(adagrad)
        theta -= temp
        acc_history[i] = accuracy(predict(X,theta),y)

    return theta, acc_history, adagrad


# In[10]:

def featurescaling(X):
    import numpy as np
    X = np.matrix(X)
    normX = np.matrix(X, dtype=float)
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        normX[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return normX, mu, sigma


# In[11]:

normX = featurescaling(X[:,1:])
X[:,1:]=normX[0]


# In[12]:

theta = np.zeros(shape=(X.shape[1],1))


# In[13]:

result = gradientdescent(X,y,theta,0.05,5000,0)
finaltheta = result[0]


# In[29]:

from model import prediction

# In[30]:

predict = prediction(finaltheta,normX[1],normX[2])


# In[37]:

import pickle
modelname = modelname+".pkl"
with open(modelname,'wb') as output:
    pickle.dump(predict,output)

