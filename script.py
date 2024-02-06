import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

# tsne = TSNE(n_components=2, random_state=0)
# X_2d = tsne.fit_transform(X)

# plt.figure(figsize=(15,8))
# plt.scatter(X_2d[:,0], X_2d[:,1], c=Y)
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

scaled = {
    "scaled": (X_train_scaled, X_test_scaled),
    "non-scaled": (X_train, X_test)
}

for isScaled, data in scaled.items():
    for name, model in models.items():
        model.fit(data[0], Y_train)
        y_pred = model.predict(data[1])
        accuracy = accuracy_score(Y_test, y_pred)
        print(f"Model {name} with {isScaled} data has accuracy score: {accuracy:.4f}")