#random forest
#knn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("stock_and_sent.csv")

train, test = train_test_split(df, test_size=0.3)
x_columns = ["Open", "High", "Low", "Volume", "Likes", "Retweets", "Positive", "Neutral", "Negative", "Composite"]
y_column = ["Close"]


# Create the knn model.
# Look at the five closest neighbors.

rff = RandomForestRegressor()
# Fit the model on the training data.
rff.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = rff.predict(test[x_columns])
predictions = predictions.reshape(142, 1)

# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)