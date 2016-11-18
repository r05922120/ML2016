
# coding: utf-8
import sys
DataDIR = sys.argv[1]
modelname = sys.argv[2]
predictPath = sys.argv[3]
# In[8]:

import pickle
import numpy as np


# In[9]:
testPath = DataDIR+"test.p"
testLoad = pickle.load(open(testPath,'rb')) 


# In[10]:

test = np.array(testLoad['data'])
X_test = np.zeros(shape=(test.shape[0], 3, 32, 32),dtype='int')
for i in range(test.shape[0]):
    X_test[i,:,:,:] = test[i].reshape((3,32,32))


# In[11]:

X_test = X_test.astype('float32')


# In[1]:

from keras.models import load_model


# In[2]:
modelPath = modelname+".h5"
model = load_model(modelPath)


# In[12]:

result = model.predict(X_test)


# In[14]:

prediction = np.zeros(shape=(X_test.shape[0],2),dtype = 'int')
prediction[:,0] = np.array(testLoad['ID'])
for i in range(X_test.shape[0]):
    prediction[i,1] = np.argmax(result[i,:])


# In[ ]:

import pandas as pd
df = pd.DataFrame(prediction,dtype='int')
df.columns=['ID','class']
df.to_csv(predictPath, index=False)

