
# coding: utf-8

# In[1]:

import numpy as np
class prediction:
    
    def __init__(self,theta,mu,sigma):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
    
    def featurescaling(self,X):
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - self.mu[i])/self.sigma[i] 
        return X
    
    def sigmod(self,z):
        z=np.matrix(z)
        return 1/(1+np.exp(-z))

    def predict(self,X):
        theta = self.theta
        X = np.hstack([np.ones(shape=(X.shape[0],1)),X])
        predict = np.matrix(np.zeros(shape=(X.shape[0],1)))
        count=0
        for i in np.array(self.sigmod(X*theta)):
            if i >=0.5:
                predict[count,0]=1
            else:
                predict[count,0]=0
            count+=1
        return predict

