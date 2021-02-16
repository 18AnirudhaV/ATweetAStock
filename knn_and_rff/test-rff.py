import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("tslaFinData.csv")
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

train, test = train_test_split(df, test_size=0.3)
x_columns = ['high',
             'low',
             'average',
             'volume',
             'notional',
             'numberOfTrades',
             'marketHigh',
             'marketLow',
             'marketAverage',
             'marketVolume',
             'marketNotional',
             'marketNumberOfTrades',
             'open',
             'marketOpen',
             'marketClose',
             'changeOverTime',
             'marketChangeOverTime']

y_column = ["close"]


# Create the knn model.
# Look at the five closest neighbors.

rff = RandomForestRegressor()
# Fit the model on the training data.
rff.fit(train[x_columns], train[y_column].values.ravel())
# Make point predictions on the test set using the fit model.
predictions = rff.predict(test[x_columns])
predictions = predictions.reshape(2796, 1)

# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)