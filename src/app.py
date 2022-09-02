import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB # cuando es texto se usa la multinomial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

url='https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv'
df = pd.read_csv(url)

df[df.select_dtypes('object').columns]=df[df.select_dtypes('object').columns].astype('string')
df['polarity']=df['polarity'].astype('category')

df['review']=df['review'].str.strip() # elimina espacios al comienzo y al final de la oracion
df['review']=df['review'].str.lower() # lleva todo a minuscula

def normalize_str(text_string):
    if text_string is not None:
        result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else:
        result=None 
    return result

df['review']=df['review'].apply(normalize_str)
df['review']=df['review'].str.replace('!','')
df['review']=df['review'].str.replace(',','')
df['review']=df['review'].str.replace('&','')
df['review']=df['review'].str.normalize('NFKC')
df['review']=df['review'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True) # elimina caracteres repetidos mas de dos veces

X=df['review']
y=df['polarity']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2007,stratify=y)

vec=CountVectorizer(stop_words='english')
X_train=vec.fit_transform(X_train).toarray()
X_test=vec.transform(X_test).toarray()

nb=MultinomialNB()
nb.fit(X_train,y_train)
print('R score', nb.score(X_train,y_train))

y_predict=nb.predict(X_test)
print(classification_report(y_test,y_predict))

import pickle
filename = '/workspace/Naive-Bayes/models/finalized_model.sav'
pickle.dump(nb, open(filename, 'wb'))