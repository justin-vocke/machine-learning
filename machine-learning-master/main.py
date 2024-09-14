import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read csv into a dataframe
house_data = pd.read_csv("house_prices.csv")
size = house_data['sqft_living']
price = house_data['price']

#machine learning handles arrays, not dataframes
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

#we use linear regression + fit() for training
model = LinearRegression()
model.fit(x,y)

#MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

#we can get the b values after the model fit

# b1
print("b1: ", model.coef_[0])

# b0
print("b0: ", model.intercept_[0])

# visualize the dataset with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("Size in Square Feet")
plt.ylabel("Price")
plt.show()
