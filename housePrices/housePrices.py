import pandas as pd
import numpy as np

# Load data
train_data_file_path = './housePrices/train.csv'
train_data = pd.read_csv(train_data_file_path)
test_data_file_path = './housePrices/test.csv'
test_data = pd.read_csv(test_data_file_path)
# Data exploration
train_data.shape
train_data.head()
train_data.describe()
train_data.columns
# Do not filter rows with missing values since every row has missing values
# house_price_data = house_price_data.dropna(axis=0)
# Choose target and features - for now features are non categorical - need to learn how to handle categorical
y = train_data.SalePrice
X = train_data.select_dtypes(include=np.number)
X.drop('SalePrice', axis=1, inplace=True)
X.shape
X.describe()
X.head()
# Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
train_model = DecisionTreeRegressor(random_state=1)
# Fit model and test for first 5 houses
train_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(train_model.predict(X.head()))
# Validate the model
from sklearn.metrics import mean_absolute_error
predicted_train = train_model.predict(X)
mean_absolute_error(y, predicted_train)

# Using non in-sample scores
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
train_model = DecisionTreeRegressor()
# Fit model
train_model.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = train_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions)) # Really bad model - mae is about 25600

# Underfitting and Overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# looks like 50 is good
# try different range

for max_leaf_nodes in [10, 30, 50, 75]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# looks like it gets better at 75, go that direction

for max_leaf_nodes in range(60, 75):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# best score is 68 or 69 nodes ... but the curve isn't as simple as low point then back up ... it wavers

# creating the final model
final_model = DecisionTreeRegressor(max_leaf_nodes = 68, random_state = 1)
final_model.fit(X, y)
test_data = test_data.select_dtypes(include=np.number)
predictions = final_model.predict(test_data)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
output.to_csv('./housePrices/submission.csv', index=False)

# try random forest model
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
train_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, train_preds)) # this is better again!

# try on test data
train_preds = forest_model.predict(test_data)