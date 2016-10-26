
# coding: utf-8

import sys
modelname=sys.argv[1]
testdata=sys.argv[2]
outputdata=sys.argv[3]
# In[2]:

import pickle
from model import a
modelname = modelname+".pkl"
with open(modelname,'rb') as input:
    predict = pickle.load(input)


# In[4]:

w = predict.w
b = predict.b


# In[5]:

import numpy as np

def sigmod(z):
    z=np.matrix(z)
    return 1/(1+np.exp(-z))
def predict(X):
    predict = np.matrix(np.zeros(shape=(X.shape[0],1)))
    count=0
    for i in X:
        if i >=0.5:
            predict[count,0]=0
        else:
            predict[count,0]=1
        count+=1
    return predict


# In[7]:

import pandas as pd
test = pd.read_csv(testdata, encoding='big5',header=None)
testx = np.matrix(test.values[:,1:])
temp =sigmod(w*(testx.T) + b)
temp = np.array(temp).reshape(600)
pred=predict(temp)


# In[8]:

output = np.zeros(shape=(pred.shape[0],2))
for i in range(pred.shape[0]):
    output[i,0]=i+1
    output[i,1]=pred[i]
df = pd.DataFrame(output,dtype='int')
df.columns=['id','label']
df.to_csv(outputdata, index=False)

