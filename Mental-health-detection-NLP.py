#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
#import re


# In[2]:


data = pd.read_csv('Downloads/REDUCED-REDDIT-DATA-10K.csv')


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.isnull().sum


# In[7]:


#REMOVING COLUMNS
data.drop(columns=['postId','date','Postno','subtopics','platform'],inplace=True)


# In[8]:


data.head()


# In[9]:


#expanding the dispay of text sms column
pd.set_option('display.max_colwidth', -1)
#using only v1 and v2 column
data= data [['post','topics']]
data.head()

#checking the count of the dependent variable
data['post'].value_counts()


# In[10]:


display(data)


# In[11]:


#library that contains punctuation
import string
string.punctuation


# In[12]:


#defining the function to remove punctuation
def remove_punctuation(post):
    text_nopunc=[c for c in post if c not in string.punctuation]
    punc=''.join(text_nopunc)
    return punc
data['clean_text'] = data['post'].apply(lambda x:remove_punctuation(x))
data.head()


# In[13]:


#Removing URL
#The http characters in the regex match the literal characters.
#\S matches any character that is not a whitespace character
#The question mark ? causes the regular expression to match 0 or 1 repetitions of the preceding character
#We then have the colon and two forward slashes :// to complete the protocol.
#data['clean_text']=data['clean_text'].str.replace("http\S+"," ")
import re
data['clean_text']=data['clean_text'].str.replace(r"http\S+"," ")
data['clean_text']=data['clean_text'].str.replace("\n","  ")
data['clean_text']=data['clean_text'].str.replace("â€™","  ")
display(data)


# In[14]:


#clean is the function provided by the cleantext library.
from cleantext import clean
clean(data['clean_text'], no_emoji=True)
display(data)


# In[15]:


#Lowercase
data['clean_text']= data['clean_text'].apply(lambda x: x.lower())
display(data)


# In[16]:


#tokenization
import re
#defining function for tokenization
def tokenization(text):
    tokens = re.split('\W+',text)
    return tokens
#applying function to the column
data['msg_tokenized']= data['clean_text'].apply(lambda x: tokenization(x))
data.head()


# In[17]:


#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words )


# In[18]:


def remove_stopwords(post): 
    tokens_without_sw = [word for word in post if not word in stop_words]
    return tokens_without_sw
data['stop_words_removed']= data['msg_tokenized'].apply(lambda x: remove_stopwords(x))
display(data)


# In[19]:


nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
w_lemmatized=lemmatizer.lemmatize(data['post'])

display(data)

#lemm = lemmatize(data['post'])
#lemm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




