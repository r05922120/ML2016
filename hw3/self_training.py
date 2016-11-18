
# coding: utf-8

# In[2]:



import sys
DataDIR = sys.argv[1]
modelname = sys.argv[2]
# In[1]:

import theano


# In[2]:

import pickle


# In[3]:

import os
os.environ["THEANO_FLAGS"] ="device=gpu0"


# In[4]:
labelPath=DataDIR+"all_label.p"
all_label = pickle.load(open(labelPath,'rb')) 


# In[5]:

type(all_label)


# In[6]:

import numpy as np


# In[7]:

all_label = np.array(all_label)


# In[8]:

nclass = all_label.shape[0]
classnum = all_label.shape[1]
all_label.shape


# In[13]:

from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Convolution2D, MaxPooling2D 
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2


# In[131]:

X_train = np.zeros(shape=(nclass*classnum, 3, 32, 32),dtype='int')
y_train = np.zeros(shape=(nclass*classnum,1),dtype='int')


# In[132]:

count=0
for i in range(nclass):
    for j in range(classnum):
        X_train[count,:,:,:]=all_label[i,j,:].reshape((3,32,32))
        y_train[count,0]=i
        count+=1


# In[133]:

Y_train = np_utils.to_categorical(y_train, nclass)


# algo2

# In[134]:

X_train = X_train.astype('float32')
X_train_norm = X_train/255


# In[25]:

model=Sequential()
model.add(Convolution2D(128,5,5,activation='relu',border_mode='same',input_shape=(3,32,32)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64,5,5,activation='relu',border_mode='same'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Convolution2D(64,3,3,activation='relu',border_mode='same'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))


# In[26]:

model.add(Flatten())
model.add(Dense(output_dim=64,init='normal',W_regularizer=l2(0.03)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(output_dim=10,W_regularizer=l2(0.01)))
model.add(Activation('linear'))
model.add(Dropout(0.2))

model.add(Dense(nclass,init='normal'))
model.add(Activation('softmax'))


# In[27]:

model.summary()


# In[28]:

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# In[29]:

model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy']) 


# In[136]:

model.fit(X_train, Y_train, batch_size=100, nb_epoch=50,shuffle=True,callbacks=[EarlyStopping(monitor='loss',patience=10,min_delta=0.1,mode='auto')])


# In[169]:

model.evaluate(X_train,Y_train)


# load unlabel data

# In[139]:

unlabelPath=DataDIR+"all_unlabel.p"
unlabelLoad = pickle.load(open(unlabelPath,'rb'))


# In[140]:

unlabelData = np.array(unlabelLoad)
unlabel_X = np.zeros(shape = (unlabelData.shape[0],3,32,32))
for i in range(unlabelData.shape[0]):
    unlabel_X[i,:,:,:] = unlabelData[i].reshape((3,32,32))


# In[141]:

origin_unlabelX = unlabel_X


# In[142]:

from scipy import stats


# In[143]:

unlabel_X = unlabel_X.astype('float32')
unlabel_X = unlabel_X/255


# # self-training


while True:
# In[171]:

    unlabel_result = model.predict(unlabel_X)


# In[172]:

    result_entropy = stats.entropy(unlabel_result.T)


# In[173]:

    index=np.asarray(np.where(result_entropy<=0.8)[0])
    index.shape
    
    if index.shape[0] < 2000:
        break

# In[174]:

    else_index=np.asarray(np.where(result_entropy>0.8)[0])
    else_index.shape


# In[175]:

    new_labelX = unlabel_X[index,:,:,:]
    new_label_result = unlabel_result[index,:]
    new_labely = np.zeros(shape=(new_labelX.shape[0],1),dtype='int')
    for i in range(new_labelX.shape[0]):
        new_labely[i,0] = np.argmax(new_label_result[i,:])
        
    new_labely=np.array(new_labely,dtype='int')
    new_labelX.shape


# In[176]:

    new_label_dum = np_utils.to_categorical(new_labely, nclass)
    new_label_dum.shape


# In[177]:

    X_final = np.vstack([X_train,new_labelX])
    Y_final = np.vstack([Y_train,new_label_dum])


    # In[178]:

    print(X_final.shape,Y_final.shape)


    # In[179]:

    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 


    # In[ ]:

    model.fit(X_final, Y_final, batch_size=100, nb_epoch=50,shuffle=True, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=5,min_delta=0.02,mode='auto')])


    # In[168]:

    model.evaluate(X_final, Y_final)


    # In[170]:

    unlabel_X = unlabel_X[else_index,:,:,:]
    Y_train = Y_final
    X_train = X_final


# # save model

# In[156]:
modelPath = modelname+".h5"
model.save(modelPath)



# In[ ]:



