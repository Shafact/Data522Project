# This project is to distinguish spam text

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O


import os
for dirname, _, filenames in os.walk('/Users/sharonzhou/Documents/GitHub/Data522/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/Users/sharonzhou/Documents/GitHub/Data522/spam.csv',encoding='latin-1')

#let's first drop the unknown columns
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df=df.rename({'v1':'target',
             'v2':'text'},axis=1)

#let's make another column i.e the length of the text
len_text=[]
for i in df['text']:
    len_text.append(len(i))

df['text_length']=len_text

# use matplotlib to make bar graph for spam message length
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
df[df['target']=='spam']['text_length'].plot(bins=35,kind='hist',color='blue',label='spam',alpha=0.5)
plt.legend()
plt.xlabel('message length')
plt.show()

plt.figure(figsize=(12,5))
df[df['target']=='ham']['text_length'].plot(bins=35,kind='hist',color='red',label='ham',alpha=0.5)
plt.legend()
plt.xlabel('message length')
plt.show()

plt.figure(figsize=(12,5))
df['target'].value_counts().plot(kind='bar',color='green',label='spam-vs-nonspam')
plt.legend()
plt.show()


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df['target']=np.where(df['target']=='spam',1,0)

spam=[]
ham=[]
spam_class=df[df['target']==1]['text']
ham_class=df[df['target']==0]['text']

def extract_ham(ham_class):
    global ham
    words = [word.lower() for word in word_tokenize(ham_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    ham=ham+words

def extract_spam(spam_class):
    global spam
    words = [word.lower() for word in word_tokenize(spam_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    spam=spam+words

spam_class.apply(extract_spam)
ham_class.apply(extract_ham )

from wordcloud import WordCloud
spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

ham_cloud=WordCloud(width=600,height=400,background_color='white').generate(" ".join(ham))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(ham_cloud)
plt.tight_layout(pad=0)
plt.show()

#top 10 spam words=
spam_words=np.array(spam)
pd.Series(spam_words).value_counts().head(n=10)

#top 10 ham words
ham_words=np.array(ham)
pd.Series(ham_words).value_counts().head(n=10)


