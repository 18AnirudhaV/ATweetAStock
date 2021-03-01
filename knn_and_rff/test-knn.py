import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("tslaMerge.csv")

train, test = train_test_split(df, test_size=0.3)
x_columns = ['volume',
             'notional',
             'numberOfTrades',
             'open',
             'likes',
             'retweets',
             'neutral',
             'negative',
             'positive',
             'composite']

y_column = ["close"]

# Create the knn model.
knn = KNeighborsRegressor()
# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])
# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)
