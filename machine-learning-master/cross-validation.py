import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# turn dataframes into arrays for ML model
x = np.array(features).reshape(-1, 3)
y = np.array(target)

# create the model using logistic regression and use 5 folds in the cross-validation
model = LogisticRegression()
predicted = cross_validate(model, x, y, cv=5)

# print the average of the 5 runs of cv
print(np.mean(predicted['test_score']))