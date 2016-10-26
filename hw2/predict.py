
# coding: utf-8

# In[ ]:

import sys
modelname = sys.argv[1]
testdata = sys.argv[2]
outputfile = sys.argv[3]


# In[1]:

import pickle
from model import prediction


# In[2]:

modelname = modelname+".pkl"
with open(modelname,'rb') as input:
    predict = pickle.load(input)


# In[9]:

import pandas as pd
import numpy as np
test = pd.read_csv(testdata, encoding='big5',header=None)
testx = np.matrix(test.values[:,1:])
testx = predict.featurescaling(testx)
pre = predict.predict(testx)


# In[10]:

output = np.zeros(shape=(pre.shape[0],2))
for i in range(pre.shape[0]):
    output[i,0]=i+1
    output[i,1]=pre[i]
df = pd.DataFrame(output,dtype='int')
df.columns=['id','label']
df.to_csv(outputfile, index=False)

