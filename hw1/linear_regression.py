
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

d = pd.read_csv("train.csv", encoding='big5')
d = d.replace('NR', 0)

def findvalues(X, name):
    import numpy as np
    output = X[np.where(X == name)[0], 3:]
    output = np.matrix(output, dtype='float')
    output = output.reshape(output.shape[0] * output.shape[1])
    return output

def monthFindValue(X, name):
    temp = findvalues(X, name)
    perMonth = int(temp.shape[1]/12)
    output = temp.reshape((12, perMonth))
    return output


# In[3]:

y = np.array([])
temp = monthFindValue(d.values, 'PM2.5')
count = 0
for i in range(12):
    y = np.append(y, np.array(temp[i, 9:]))
    
y = np.matrix(y).T
m = y.shape[0]


# In[4]:

X = np.vstack([np.ones(shape=(1, m)), np.zeros(shape=(162+162, m))]).T
X = np.matrix(X)

items = d['測項'][:18].values
testdata = np.zeros(shape=(18, 5760))
count = 0
for i in items:
    testdata[count, :] = findvalues(d.values, i)
    count+=1

count = 0
for i in range(12):
    j=0
    while j < 471:
        start = i*480+j
        X[count, 1:163] = testdata[:, start:start+9].reshape((1, 162))
        X[count, 163:] = np.square(testdata[:, start:start+9].reshape((1, 162)))
        j += 1
        count += 1


# In[5]:

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

normX = featurescaling(X[:, 1:])
X[:, 1:] = normX[0]


# In[6]:

def costfunction(X, y, theta, londa):
    import numpy as np
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = X.shape[0]

    return (np.sum(np.square((X * theta) - y)) + londa * np.sum(np.square(theta[1:]))) / (2*m)

theta = np.zeros(shape=(163+162, 1))


# In[7]:

def gradientdescent(X, y, theta, alpha, num_iters, londa):
    import numpy as np
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = X.shape[0]
    L_history = np.zeros(num_iters)
    adagrad = np.zeros(theta.shape)
    for i in range(num_iters):
        temp = X.T * (X * theta - y)
        temp[1:] += londa * theta[1:]
        temp = (alpha/m) * temp
        adagrad += np.square(temp)
        temp = temp/np.sqrt(adagrad)
        theta -= temp
        L_history[i] = costfunction(X, y, theta, londa)

    return theta, L_history, adagrad



# In[8]:

train_result=gradientdescent(X, y, theta, 0.1, 10000, 100)


# In[9]:

finaltheta = train_result[0]


# In[10]:

dtest = pd.read_csv("test_X.csv", encoding='big5', header=None)
dtest = dtest.replace('NR', 0)
output = np.matrix(dtest.values[:, 2:], dtype='float')


# In[11]:

predict = np.zeros(240)
count = 0
for i in range(240):
    temp = output[count:count+18, :].reshape((1, 162)).T
    testX = np.vstack([temp, np.square(temp)]).T
    for j in range(162+162):
        testX[0, j] = (testX[0, j] - normX[1][j]) / normX[2][j]
    testX = np.vstack([1, testX.T]).T
    predict[i] = testX * finaltheta
    count+=18

intpredict = np.array(predict, dtype='int')


# In[12]:

sample = pd.read_csv("sampleSubmission.csv", encoding='big5')
for i in range(240):
    sample.set_value(i, 'value', intpredict[i])
sample.to_csv("linear_regression.csv", index=False)

