import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = pd.read_csv("credit_data.csv")
features = data[["income", "age", "loan"]]
target = data.default
#print(data[["income"]].max())
X = np.array(features).reshape(-1, 3)
y= np.array(target)

# want to normalize the data for knn using min-max normalization
X = preprocessing.MinMaxScaler().fit_transform(X)

feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=.3)

model = KNeighborsClassifier(n_neighbors=32)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

cross_valid_scores = []

# find the optimal number of neighbors between 1 and 100
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    cross_valid_scores.append(scores.mean())

print("Optimal k value with cross-validation: ", np.argmax(cross_valid_scores))

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

