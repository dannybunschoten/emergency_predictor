import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

data.dropna(subset=['keyword'], inplace=True)
texts = data['text'].apply(lambda x: simple_preprocess(x))
preprocessedGenre = data['keyword'].apply(lambda x: simple_preprocess(x))


model_genre = Word2Vec(sentences=preprocessedGenre, vector_size=100, window=5, min_count=1, workers=4)
model_tweet_text = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

import numpy as np

def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.key_to_index]
    if not doc:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)

X = np.hstack((np.array([document_vector(text, model_tweet_text) for text in texts]), np.array([document_vector(genre, model_genre) for genre in preprocessedGenre])))
# X = np.array([document_vector(text, model_tweet_text) for text in texts])
Y = data['target']

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(15,8))
plt.scatter(X_2d[:,0], X_2d[:,1], c=Y)
plt.show()