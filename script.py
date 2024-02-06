import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
submission = pd.read_csv('test.csv')

# data.dropna(subset=['keyword'], inplace=True)
texts = data['text'].apply(lambda x: simple_preprocess(x))
texts_submission = submission['text'].apply(lambda x: simple_preprocess(x))
# preprocessedGenre = data['keyword'].apply(lambda x: simple_preprocess(x))

genres = data[['keyword']]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
genresEncoded = encoder.fit_transform(genres)
submissionEncoded = encoder.transform(submission[['keyword']])

# model_genre = Word2Vec(sentences=preprocessedGenre, vector_size=100, window=5, min_count=1, workers=4)
model_tweet_text = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)

def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.key_to_index]
    if not doc:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)

# X = np.hstack((np.array([document_vector(text, model_tweet_text) for text in texts]), np.array([document_vector(genre, model_genre) for genre in preprocessedGenre])))
# X = np.array([document_vector(text, model_tweet_text) for text in texts])
X = np.hstack((genresEncoded, np.array([document_vector(text, model_tweet_text) for text in texts])))
X_submission = np.hstack((submissionEncoded, np.array([document_vector(text, model_tweet_text) for text in texts_submission])))

Y = data['target']

# tsne = TSNE(n_components=2, random_state=0)
# X_2d = tsne.fit_transform(X)

# plt.figure(figsize=(15,8))
# plt.scatter(X_2d[:,0], X_2d[:,1], c=Y)
# plt.show()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)
X_submission = scaler.transform(X_submission)

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Random Forest": RandomForestClassifier(),
#     "SVM": SVC(),
#     "KNN": KNeighborsClassifier()
# }

# scaled = {
#     "scaled": (X_train_scaled, X_test_scaled),
#     "non-scaled": (X_train, X_test)
# }

# for isScaled, data in scaled.items():
#     for name, model in models.items():
#         model.fit(data[0], Y_train)
#         y_pred = model.predict(data[1])
#         accuracy = accuracy_score(Y_test, y_pred)
#         print(f"Model {name} with {isScaled} data has accuracy score: {accuracy:.4f}")

# After consideration of these values, we will proceed with logistic regression on scaled data
        
# logistic_regression_model = LogisticRegression(max_iter=10000)

# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
#     'penalty': ['l1', 'l2', 'elasticnet']
# }

# grid_search = GridSearchCV(logistic_regression_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train_scaled, Y_train)

# print(f"Best paramters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")
        
trained_model = LogisticRegression(penalty='l1', C=10, solver='saga', max_iter=10000)
trained_model = trained_model.fit(X=X_train_scaled, y=Y)

submission_predicted = trained_model.predict(X_submission)
submission_final = pd.DataFrame(
    {
        'id': submission['id'], 
        'target': submission_predicted
    }
)

submission_final.to_csv('submissions.csv', index=False)