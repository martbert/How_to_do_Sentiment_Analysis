
# coding: utf-8

# In[130]:

# TFlearn helper to setup neural networks
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences

# Tools to scrape the web
import re, urllib, requests, os
from collections import Counter
from bs4 import BeautifulSoup, SoupStrainer

# Data wrangling modules
import numpy as np
import pandas as pd


# In[61]:

# Function to scrape the IGN website for verdicts

def get_review_verdict(game_url):
    url = 'http://ca.ign.com%s' % (game_url)
    response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})

    # parse the search page using SoupStrainer and lxml
    soup = BeautifulSoup(response.content, 'lxml')

    # Get the review link
    rev_tag = soup.find(name="a", class_="reviewLink", string=re.compile('Review'))
    try:
        rev_url = rev_tag.attrs['href']

        # Extract the verdict of the review
        response = requests.get(rev_url, headers={'User-agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'lxml')
        verdict_tag = soup.find(name='div', class_='articleSubHeader', string=re.compile('The Verdict'))

        verdict = verdict_tag.next_sibling.next_sibling.next_sibling
        verdict_str = ''.join([s for s in verdict.strings])
    except:
        verdict_str = None

    return verdict_str


# In[63]:

# Read ign data
# If the dataset with appended reviews exists read it, if not query 

if os.path.exists('ign_reviews.csv'):
    ignd = pd.read_csv('ign_reviews.csv', index_col=0)
else:
    ignd = pd.read_csv('ign.csv', index_col=0)
    
    # Fill in the reviews (this takes a while)
    ignd['Reviews'] = ignd['url'].map(lambda x: get_review_verdict(x))
    
    # Save to file for future use
    ignd.to_csv('ign_reviews.csv')


# In[64]:

# Loop and create bag of words using a Counter
# Some reviews were most probably not successfully retrieved so they need to be filtered out

bw = Counter()
mask = ignd['Reviews'].isnull() == False

char_to_sub = '|'.join(['\(', '\)', ',', '\.', '\?', '!', ])
pat = re.compile(char_to_sub) 

for rev in ignd.loc[mask, 'Reviews']:
    rev = rev.lower()
    rev = pat.sub('', rev)
    words = rev.split()
    for w in words:
        bw[w] += 1


# In[67]:

# Build a corpus out of the 10000 most frequent words
# The corpus is a dictionary with index association

corpus = {w[0]:idx+1 for w,idx in zip(bw.most_common(10000), range(1000))}


# In[137]:

# Transform every review in a sequence of numbers associated with the words

# Helper function to convert a text (words) to a sequence of indices
def text_to_seq(text):
    text.lower()
    text = pat.sub('', text)
    words = text.split()
    idx = [corpus.get(w, -1) for w in words]
    return [i for i in idx if i != -1]
    
wseq = []
lengths = []
for rev in ignd.loc[mask, 'Reviews']:
    idx = text_to_seq(rev)
    lengths.append(len(idx))
    wseq.append(idx)


# In[116]:

# Get the scores associated with the reviews

scores = ignd.score[mask] / 10

mask1 = scores <= 1/3
mask2 = (scores > 1/3) & (scores <= 2/3)
mask3 = scores > 2/3

scores[mask1] = 0
scores[mask2] = 1
scores[mask3] = 2


# In[118]:

# Prepare data for the neural network

# Sequence padding
X = pad_sequences(wseq, maxlen=max(lengths), value=0.)
# Converting labels to binary vectors
Y = to_categorical(scores, nb_classes=3)


# In[120]:

# Take 90% for training and 10% for testing purposes

idx = int(0.9 * X.shape[0])
trainX, testX = X[:idx], X[idx:]
trainY, testY = Y[:idx], Y[idx:]


# In[126]:

trainX.shape


# In[131]:

# Network building function

def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    net = tflearn.input_data([None, X.shape[1]])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 3, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    
    return net


# In[132]:

# Build the network and train

net = build_model()
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          n_epoch=50, batch_size=32)

# Save model
model.save('sentiment_analysis_rnn.tflearn')



