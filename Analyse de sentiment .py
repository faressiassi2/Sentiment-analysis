#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# first we just use the word embedding


# In[10]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re # regular expression
import string
import nltk
from nltk.util import pr
from nltk.corpus import stopwords


# In[2]:


data = pd.read_csv('Tweets.csv')
data.head()


# In[3]:


data = data[['text', 'airline_sentiment']]
data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[8]:


data['airline_sentiment'].value_counts()


# In[7]:


# Data preparation


# In[ ]:


#Nettoyage des données
#La première chose que nous allons faire est de supprimer les mots vides. 
#Ces mots n'ont aucune valeur pour prédire le sentiment. 
#De plus, comme nous voulons construire un modèle qui peut également être utilisé pour d'autres compagnies aériennes, 
#nous supprimons les mentions.


# In[11]:


stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword = set(stopwords.words('english'))


# In[12]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


# In[13]:


data["text"] = data["text"].apply(clean)


# In[14]:


data


# In[17]:


X_train = data['text']


# In[19]:


y_train = data['airline_sentiment']


# In[20]:


x_train, X_test, Y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


# In[42]:


x_train[0]


# In[48]:


X_test[8986]


# In[36]:


le = LabelEncoder()
y_train_le = le.fit_transform(Y_train)


# In[34]:


len(x_train)


# In[37]:


len(y_train_le)


# In[54]:


y_test_le = le.transform(y_test)


# In[78]:


len(X_test)


# In[77]:


len(y_test_le)


# In[15]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[16]:


#we define the hyperparametre:
vocab_size = 1000
embedding_dim = 16
max_length = 20


# In[21]:


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOF>")
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index


# In[22]:


word_index


# In[23]:


sequences = tokenizer.texts_to_sequences(x_train)


# In[40]:


sequences[0]


# In[39]:


sequences_test = tokenizer.texts_to_sequences(X_test)


# In[50]:


sequences_test[0]


# In[24]:


padded = pad_sequences(sequences,truncating='post',maxlen=max_length)


# In[51]:


padded[0]


# In[52]:


padded_test = pad_sequences(sequences_test,truncating='post',maxlen=max_length)


# In[89]:


model = keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(512)),
    keras.layers.RepeatVector(3),
    keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
    keras.layers.Flatten(),
    #keras.layers.GlobalAveragePooling1D(),
    #keras.layers.Dense(16,activation='relu'),
    #keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(3,activation='softmax')
])


# In[90]:


model.summary()


# In[91]:


model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)


# In[93]:


history = model.fit(
    padded,
    y_train_le,
    epochs = 10,
    batch_size = 256,
    validation_data=(padded_test, y_test_le)
)


# In[ ]:


#Evaluate the model on the test data using `evaluate`: 
score = model.evaluate(padded_test, y_test_le, verbose=0)
print("test loss, test acc:", score)


# In[61]:


# Plot the loss and accuracy curves for training and validation:

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs,acc,label='Training accuracy')
plt.plot(epochs,val_acc,label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,label='Training Loss')
plt.plot(epochs,val_loss,label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


# Handling overfitting
# Now, we can try to do something about the overfitting. There are different options to do that.
# Option 1: reduce the network's size by removing layers or reducing the number of hidden elements in the layers
# Option 2: add regularization, which comes down to adding a cost to the loss function for large weights
# Option 3: adding dropout layers, which will randomly remove certain features by setting them to zero

