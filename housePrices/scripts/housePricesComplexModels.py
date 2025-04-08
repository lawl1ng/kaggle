import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
import os

# Get train and test data
print(os.getcwd())
train = pd.read_csv('./housePrices/data/train.csv', index_col=0)
test = pd.read_csv('./housePrices/data/test.csv', index_col=0)

print("train: ", train.shape)
print("test: ", test.shape)
train.head()

# Some columns have a lot of NAN (Not a Number). We'll investigate this later
# Concatenate the train and test data. Makes it more convenient for preprocessing the data later
X = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
#X = train.drop('SalePrice', axis=1)
y = train[['SalePrice']]

X.info()
X.shape

# Isolating the numerical and categorical columns since different visualisation techniques will be used on them
numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
numeric_.columns

discrete_num_var = ['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

continuous_num_var = []
for i in numeric_.columns:
    if i not in discrete_num_var:
        continuous_num_var.append(i)

cat_train = X.select_dtypes(include=['object']).copy()
cat_train['MSSubClass'] = X['MSSubClass'] # MSSubClass is nominal
cat_train.columns

# Univariate analysis - looking at the distribution of numerical columns and mean, median, mode. Use distribution plot to visualise data distribution. Can use boxplots as well. Can look for outliers to filter out later in the preprocessing step.

fig = plt.figure(figsize=(18,16))
for index,col in enumerate(continuous_num_var):
    plt.subplot(6,4,index+1)
    sns.distplot(numeric_.loc[:,col].dropna(), kde=False)
fig.tight_layout(pad=2.0)
plt.show()

# Can see variables with mostly 1 value are BsmtFinSF2, LowQualFinSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal
# All these features are highly skewed, with mostly 0s. Having alot of 0s in the distribution doesnt really add info for predicting Housing Price. Hence we will remove them during our preprocessing step

fig = plt.figure(figsize=(14,15))
for index,col in enumerate(continuous_num_var):
    plt.subplot(6,4,index+1)
    sns.boxplot(y=col, data=numeric_.dropna())
fig.tight_layout(pad=3.0)
plt.show()

fig = plt.figure(figsize=(20,15))
for index,col in enumerate(discrete_num_var):
    plt.subplot(5,3,index+1)
    sns.countplot(x=col, data=numeric_.dropna(), palette='deep')
fig.tight_layout(pad=2.0)
plt.show()

# Categorical features - use countplots to visualise the count of each distinct value within each feature. Some categorical features (columns/data) mainly have one value, which does not add any useful information. So, they'll be removed later.

fig = plt.figure(figsize=(18,20))
for index in range(len(cat_train.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_train.iloc[:,index], data=cat_train.dropna(), palette='deep')
    plt.xticks(rotation=90)
fig.tight_layout(pad=3.0)
plt.show()

# not really readable this previous output.. could work on that.
# Looking at bivariate analysis now.
# Look at a correlation matrix, investigate possibility of multicollinearity (when 2 or more independent variables are highly correlated)

plt.figure(figsize=(14,12))
correlation = numeric_.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')
plt.show()

# Highly Correlated variables:
#   -GarageYrBlt and YearBuilt
#   -TotRmsAbvGrd and GrLivArea
#   -1stFlrSF and TotalBsmtSF
#   -GarageArea and GarageCars
# We'll use this later, when removing highly correlated features to avoid performance loss in the model

# Find correlation of SalePrice with numeric variables
numeric_train = train.select_dtypes(exclude=['object'])
correlation = numeric_train.corr()
correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False)

# Creating Scatterplots. Doesn't provide evidence of strength of linear relationship, but can help visualise any sort of relationship the correlation matrix could not calculate. E.g. Quadratic, Exponential

fig = plt.figure(figsize=(20,20))
for index in range(len(numeric_train.columns)):
    plt.subplot(8,5,index+1)
    sns.scatterplot(x=numeric_train.iloc[:,index], y='SalePrice', data=numeric_train.dropna())
fig.tight_layout(pad=3.0)
plt.show()

# Data Processing - based off the initial analysis and visualisation. Provide clean and error-free data for the model to train on.
# Remove reduntant features (columns)
# Deal with outliers
# Fill missing values

# Remove columns with multicollinearity
X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'], axis=1, inplace=True)

# Remove columns with alot of missing values e.g. Alley
plt.figure(figsize=(25,8))
plt.title('Number of missing rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()
missing_count.columns = ['features','sum']
sns.barplot(x='features',y='sum', data = missing_count, palette='deep', legend=False)
plt.show()

X.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)

# Drop columns that don't have a linear relationship with SalePrice. MoSold and YrSold don't have any impact on the price of the house sold...
fig,axes = plt.subplots(1,2, figsize=(15,5))
sns.regplot(x=numeric_train['MoSold'], y='SalePrice', data=numeric_train, ax = axes[0], line_kws={'color':'black'})
sns.regplot(x=numeric_train['YrSold'], y='SalePrice', data=numeric_train, ax = axes[1],line_kws={'color':'black'})
fig.tight_layout(pad=2.0)
plt.show()

correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False).tail(20) # MoSold and YrSold have low correlation

X.drop(['MoSold','YrSold'], axis=1, inplace=True)

# Remove features which have mostly just 1 value. Set user defined threshold of 96% - if column has more than 96% of the same value, get rid.
cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)

overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)

num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)

overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)

print("Categorical Features with >96% of the same value: ",overfit_cat)
print("Numerical Features with >96% of the same value: ",overfit_num)

# Dealing with outliers
# See from boxplots which values have extreme outliers - remove the outliers based on threshold value
out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea']
fig = plt.figure(figsize=(20,5))
for index,col in enumerate(out_col):
    plt.subplot(1,5,index+1)
    sns.boxplot(y=col, data=X)
fig.tight_layout(pad=1.5)
plt.show()

train = train.drop(train[train['LotFrontage'] > 200].index)
train = train.drop(train[train['LotArea'] > 100000].index)
train = train.drop(train[train['BsmtFinSF1'] > 4000].index)
train = train.drop(train[train['TotalBsmtSF'] > 5000].index)
train = train.drop(train[train['GrLivArea'] > 4000].index)

# Fill missing values
# ML model is unable to deal with missing values, need to deal with them based on columns.
pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(15)

cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1',
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",
       "HeatingQC",'ExterQual','ExterCond']

X[cat] = X[cat].fillna("NA")

#categorical
cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))

# Numerical - using mean where there is not much not variation between neighbourhoods. Grouping by neighbourhoods where there's a lot of variation i.e. in LotFrontage and GarageArea
print("Mean of LotFrontage: ", X['LotFrontage'].mean())
print("Mean of GarageArea: ", X['GarageArea'].mean())

neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')
neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')

fig, axes = plt.subplots(1,2,figsize=(22,8))
axes[0].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='LotFrontage_mean', data=neigh_lot, ax=axes[0])
axes[1].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1])
plt.show()

#for correlated relationship
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#numerical
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
X[cont] = X[cont] = X[cont].fillna(X[cont].mean())

# Changing data type of MSSubClass
X['MSSubClass'] = X['MSSubClass'].apply(str)

# Mapping Ordinal Features
# Some columns have strings representing quality or condition - mapping these to a value (low to high representing low to high quality)
ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}

ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']

for col in ord_col:
    X[col] = X[col].map(ordinal_map)

fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)

X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)
X['Fence'] = X['Fence'].map(fence_map)

# Can now move onto Feature Engineering after removing outliers, highly correlated features and imputing missing values. Feature engineering is adding information for the model to train on.

#Based on the current feature we have, the first additional featuire we can add would be TotalLot, which sums up both the LotFrontage and LotArea to identify the total area of land available as lot. We can also calculate the total number of surface area of the house, TotalSF by adding the area from basement and 2nd floor. TotalBath can also be used to tell us in total how many bathrooms are there in the house. We can also add all the different types of porches around the house and generalise into a total porch area, TotalPorch.
#TotalLot = LotFrontage + LotArea
#TotalSF = TotalBsmtSF + 2ndFlrSF
#TotalBath = FullBath + HalfBath
#TotalPorch = OpenPorchSF + EnclosedPorch + ScreenPorch
#TotalBsmtFin = BsmtFinSF1 + BsmtFinSF2
X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']

# Also create binary columns
column = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']

for col in column:
    col_name = col+'_bin'
    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)

# Convert categorical to numerical - ml only learns from numerical data..(?)
X = pd.get_dummies(X)

# SalePrice Distribution
plt.figure(figsize=(10,6))
plt.title("Before transformation of SalePrice")
dist = sns.distplot(train['SalePrice'],norm_hist=False)
plt.show()

# Positively skewed distribution, tail on the right is longer than the tail on the left. Use log transformation to reduce skewness.
plt.figure(figsize=(10,6))
plt.title("After transformation of SalePrice")
dist = sns.distplot(np.log(train['SalePrice']),norm_hist=False)
plt.show()

y["SalePrice"] = np.log(y['SalePrice'])

# Satisfied with final data, proceed to modelling.
# This consists of scaling the data for better optimisation
# and different ensembling methods for predicting
# there's also some hyperparameter tuning - but that's covered in a separate notebook (by AQX)

x = X.loc[train.index]
y = y.loc[train.index]
test = X.loc[test.index]

# RobustScaler removes median and scales data according to IQR. Good for data with a lot of outliers
# Doing this on both train and test data opens the set up to data leakage

from sklearn.preprocessing import RobustScaler

cols = x.select_dtypes(np.number).columns
transformer = RobustScaler().fit(x[cols])
x[cols] = transformer.transform(x[cols])
#test[cols] = transformer.transform(test[cols])

# Ensemble algorithms
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=0)

# Using decision tree regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

dtr_model = DecisionTreeRegressor()
dtr_model.fit(train_X, train_y)
val_predictions = dtr_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [200, 250, 300, 350]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

final_dtr_model = DecisionTreeRegressor(max_leaf_nodes=300, random_state=1)
final_dtr_model.fit(x, y)
predictions = final_dtr_model.predict(test)
output = pd.DataFrame({'Id': test.index, 'SalePrice': predictions})
output.to_csv('./housePrices/submissions/submissionDTR1.csv', index=False)
# Output does not look right. Submit and see how accurate it is. Not accurate at all.
# At least learned high compute power needed and how to do a lot of EDA.

# Commenting out all of the following because it's too computer intensive for the hardware I currently have.
# Catboost, XGBoost and LGBM all require an i7 CPU, currently I've i5
# # Focus on boosting (from bagging, boosting and stacking)
# # Boosting works on a class of weak learners, improving them into strong learners.
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from xgboost import XGBRegressor
# from sklearn import ensemble
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import cross_val_score
# from catboost import CatBoostRegressor

# # xgboost is extreme gradient boost - uses the gradient boosting framework. Gradient descent algo is employed to minimise errors in the sequential model. It improves on the gradient boosting framework with faster execution speed and improved faster performance.
# # Tried XGB but it took an enormous amount of CPU, at least more than I had...
# xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')

# # XGBoost HyperParameter Tuning
# from sklearn.model_selection import RandomizedSearchCV

# param_lst = {
#     'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
#     'n_estimators' : [100, 500, 1000, 2000, 3000],
#     'max_depth' : [3, 6, 9],
#     'min_child_weight' : [1, 5, 10, 20],
#     'reg_alpha' : [0.001, 0.01, 0.1],
#     'reg_lambda' : [0.001, 0.01, 0.1]
# }

# xgb_reg = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst,
#                               n_iter = 100, scoring = 'neg_root_mean_squared_error',
#                               cv = 5)

# xgb_search = xgb_reg.fit(X_train, y_train)
# # Haven't ran this as it was taking a very long time...

# # XGB with tune hyperparameters
# best_param = xgb_search.best_params_
# xgb = XGBRegressor(**best_param)

# # LightGBM is another gradient boosting framework developed by Microsoft based on decision tree algo. Faster training speed and higher efficiency. Lower mem usage, better accuracy, support of parallel learning, capable of handling large-scale data.
# # Splits the tree leaf wise as opposed to level or depth wise... not sure what this means

# lgbm = LGBMRegressor(boosting_type='gbdt',objective='regression', max_depth=-1,
#                     lambda_l1=0.0001, lambda_l2=0, learning_rate=0.1,
#                     n_estimators=100, max_bin=200, min_child_samples=20,
#                     bagging_fraction=0.75, bagging_freq=5,
#                     bagging_seed=7, feature_fraction=0.8,
#                     feature_fraction_seed=7, verbose=-1)

# # LBGM Hyperparameter tuning
# # it looks like these are quite computer intensive...
# # I've an i5 (10th generation) cpu, and while it can handle building ml models, an i7 would be better.
# # CPU specs include 4 cores and 8 threads, 6MB Cache
# # 64GB of RAM
# # Apparently i5 is a starting point for ml when working with:
# # small to medium sized datasets
# # simpler algos like linear regression or decision trees
# # data preprocessing and cleaning tasks
# # basic programming objects
# param_lst = {
#     'max_depth' : [2, 5, 8, 10],
#     'learning_rate' : [0.001, 0.01, 0.1, 0.2],
#     'n_estimators' : [100, 300, 500, 1000, 1500],
#     'lambda_l1' : [0.0001, 0.001, 0.01],
#     'lambda_l2' : [0, 0.0001, 0.001, 0.01],
#     'feature_fraction' : [0.4, 0.6, 0.8],
#     'min_child_samples' : [5, 10, 20, 25]
# }

# lightgbm = RandomizedSearchCV(estimator = lgbm, param_distributions = param_lst,
#                               n_iter = 100, scoring = 'neg_root_mean_squared_error',
#                               cv = 5)

# lightgbm_search = lightgbm.fit(X_train, y_train)
# # Took a long time to run, but did run...

# # LightBGM with tuned hyperparameters
# best_param = lightgbm_search.best_params_
# lgbm = LGBMRegressor(**best_param)

# # catboost - another alternative gradient boosting framework developed by Yandex. Category boosting. Can deal with categorical features. Main point is it has features which fight the prediction shift caused by a certain target leakage present in all existing implementations of gradient boosting algos.

# cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')

# param_lst = {
#     'n_estimators' : [100, 300, 500, 1000, 1300, 1600],
#     'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
#     'l2_leaf_reg' : [0.001, 0.01, 0.1],
#     'random_strength' : [0.25, 0.5 ,1],
#     'max_depth' : [3, 6, 9],
#     'min_child_samples' : [2, 5, 10, 15, 20],
#     'rsm' : [0.5, 0.7, 0.9],

# }

# catboost = RandomizedSearchCV(estimator = cb, param_distributions = param_lst, n_iter = 100, scoring = 'neg_root_mean_squared_error', cv = 5)

# catboost_search = catboost.fit(X_train, y_train)
# # Taking >1hr to run

# # CatBoost with tuned hyperparams
# best_param = catboost_search.best_params_
# cb = CatBoostRegressor(logging_level='Silent', **best_param)

# # Training and Evaluation
# def mean_cross_val(model, X, y):
#     score = cross_val_score(model, X, y, cv=5)
#     mean = score.mean()
#     return mean

# cb.fit(X_train, y_train)
# preds = cb.predict(X_val)
# preds_test_cb = cb.predict(test)
# mae_cb = mean_absolute_error(y_val, preds)
# rmse_cb = np.sqrt(mean_squared_error(y_val, preds))
# score_cb = cb.score(X_val, y_val)
# cv_cb = mean_cross_val(cb, x, y)


# xgb.fit(X_train, y_train)
# preds = xgb.predict(X_val)
# preds_test_xgb = xgb.predict(test)
# mae_xgb = mean_absolute_error(y_val, preds)
# rmse_xgb = np.sqrt(mean_squared_error(y_val, preds))
# score_xgb = xgb.score(X_val, y_val)
# cv_xgb = mean_cross_val(xgb, x, y)


# lgbm.fit(X_train, y_train)
# preds = lgbm.predict(X_val)
# preds_test_lgbm = lgbm.predict(test)
# mae_lgbm = mean_absolute_error(y_val, preds)
# rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds))
# score_lgbm = lgbm.score(X_val, y_val)
# cv_lgbm = mean_cross_val(lgbm, x, y)

# # Model performances
# model_performances = pd.DataFrame({
#     "Model" : ["XGBoost", "LGBM", "CatBoost"],
#     "CV(5)" : [str(cv_xgb)[0:5], str(cv_lgbm)[0:5], str(cv_cb)[0:5]],
#     "MAE" : [str(mae_xgb)[0:5], str(mae_lgbm)[0:5], str(mae_cb)[0:5]],
#     "RMSE" : [str(rmse_xgb)[0:5], str(rmse_lgbm)[0:5], str(rmse_cb)[0:5]],
#     "Score" : [str(score_xgb)[0:5], str(score_lgbm)[0:5], str(score_cb)[0:5]]
# })

# print("Sorted by Score:")
# print(model_performances.sort_values(by="Score", ascending=False))

# def blend_models_predict(X, b, c, d):
#     return ((b* xgb.predict(X)) + (c * lgbm.predict(X)) + (d * cb.predict(X)))

# subm = np.exp(blend_models_predict(test, 0.4, 0.3, 0.3))