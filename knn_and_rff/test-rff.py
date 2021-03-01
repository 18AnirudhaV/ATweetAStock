import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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


# Create the rff model.
rff = RandomForestRegressor()
# Fit the model on the training data.
rff.fit(train[x_columns], train[y_column].values.ravel())
# Make point predictions on the test set using the fit model.
predictions = rff.predict(test[x_columns])
predictions = predictions.reshape(1738, 1)
# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)
