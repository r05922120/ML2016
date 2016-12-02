
# coding: utf-8



import sys
DataDIR = sys.argv[1]
outputname = sys.argv[2]
# In[18]:

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# # read file

# In[2]:
title=pd.read_csv(DataDIR+"title_StackOverflow.txt",sep='\n',header=None,encoding="utf-8")


# In[3]:

sentence=title.values
sentences=list(sentence.reshape(sentence.shape[0]))


# # segment

# In[4]:

import unicodedata
import re
for i in range(len(sentences)):
    sentences[i] = re.sub('[0-9]+', '',unicodedata.normalize('NFKC', sentences[i]))


# In[5]:

stopwords_list=stopwords.words('english')


# In[6]:

tokenizer = RegexpTokenizer(r'\w+')


# In[7]:

Corpus = list()
for i in sentences:
    words=tokenizer.tokenize(i.lower())
    Corpus.append([word for word in words if word not in stopwords_list])


# In[8]:

title_doc = list()
for i in Corpus:
    title_doc.append(' '.join(i))


# # tfidf vectorize

# In[9]:

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from nltk.corpus import stopwords
totalvec =TfidfVectorizer(encoding='utf-8',max_df=0.5,min_df=2)
total_tfidf=totalvec.fit_transform(title_doc)


# # LSA

# In[10]:

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


# In[11]:

svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(total_tfidf)


# # clustering

# In[12]:

from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


# In[14]:

km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1).fit(X)


# # submit

# In[15]:

check_index = pd.read_csv(DataDIR+"check_index.csv")


# In[16]:

x_ID=check_index['x_ID'].values
y_ID=check_index['y_ID'].values


# In[19]:

ans=np.zeros(shape=(x_ID.shape[0],1),dtype='int')
for i in range(x_ID.shape[0]):
    if km.labels_[x_ID[i]]==km.labels_[y_ID[i]]:
        ans[i,0]=1
    else:
        ans[i,0]=0


# In[20]:

ID = check_index['ID'].values.reshape((x_ID.shape[0],1))


# In[21]:

output = np.hstack([ID,ans])
output_df = pd.DataFrame(output)
output_df.columns=['ID','Ans']


# In[22]:

output_df.to_csv(outputname,index=False)

