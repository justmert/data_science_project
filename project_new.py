# %%
# In this project, we are going to do data handling, exploratory data analysis on train.csv
# and make house price prediction based on test.csv from the Ames Housing dataset.

# %%
# importing necessary libraries
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %matplotlib inline
# %%
# Reading the dataset from the csv
df_train = pd.read_csv("ames_housing/train.csv")
df_test = pd.read_csv("ames_housing/test.csv")
# %%
# Quick glance over the train set
df_train.head(5)
# %%
# Printing columns
print(df_train.columns)
print("\nThere are {} columns in the dataset".format(len(df_train.columns)))
# We can see that there a lot of features ( attributes ) in the dataset
# %%
# Let's see the non-null counts and dtypes of the columns/attributes
df_train.info()
# %%
# Describing the train set
df_train.describe()
# %%
# We are going to drop 'Id' columns since it is unnecessary for the prediction process
df_train = df_train.drop("Id", axis=1)
df_test = df_test.drop("Id", axis=1)
# %%
# Describing the our target variable 'SalePrice'
df_train['SalePrice'].describe()
# %%
# Plotting distribution of the SalePrice
sns.distplot(df_train['SalePrice'])
# %%
# correlation heatmap of the attributes
sns.set(font_scale=1.5)
corrmat = df_train.corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, vmax=.8, square=True, ax=ax)
# %%
# Sorted correlation of the SalePrice. This way, we can detect which attributes
# are linear relationship with SalePrice
corrmat.sort_values(['SalePrice'], ascending=False, inplace=True)
corrmat['SalePrice']  # the most important features relative to target
# %%
# Now we will do bivariate analysis with SalePrice and most correlated attributes with it
# GrLivArea and SalePrice Scatter Plot
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train)
# %%
# TotalBsmtSF and GrLivArea Scatter Plot
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df_train)
# %%
# TotalBsmtSF and GrLivArea Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='OverallQual', y="SalePrice", data=df_train, ax=ax)
# %%
# YearBuilt and GrLivArea Scatter Plot
fig, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='YearBuilt', y="SalePrice", data=df_train, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# %%
# We are creating annotated correlations heatmap between attributes
k = 10
sns.set(font_scale=1)
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
            'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# %%
# Plotting pairwise relationship among attributes
cols = ['SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
# %%
# Let's remember the scatter plot of GrLivArea and SalePrice
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train)
# %%
# As we can see from GrLivArea with SalePrice with scatter plot that GrLivArea could had outliers.
# Outliers are trouble that we should be aware of. Because outliers can be markedly affect our models
# and can be a valuable source of information, providing us insights about spesific behaviours.
# The two values with bigger ‘GrLivArea’ seem strange and they are not following the crowd.
# We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price.
# I'm not sure about this but I'm quite confident that these two points are not representative of the typical case.
# Therefore, we had defined them as outliers and deleted them

df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (
    df_train['SalePrice'] < 300000)].index)
df_train = df_train.drop(df_train[(df_train['OverallQual'] < 5) & (
    df_train['SalePrice'] > 200000)].index)
# %%
# After deleting outliers, GrLivArea with SalePrice with scatter plot becomes this
fig, ax = plt.subplots()
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train)
# %%
# Let's remember the distribution of SalePrice
sns.distplot(df_train['SalePrice'])
# %%
# Printing Skewness and Kurtosis of the SalePrice
print('Skewness of SalePrice: {0}'.format(df_train['SalePrice'].skew()))
print('Kurtosis of SalePrice: {0}'.format(df_train['SalePrice'].kurt()))
# %%
# The SalePrice is skewed to the right. This is a problem because most ML models don't do well with non-normally distributed data.
# We can apply a log(1+x) tranform to fix the skew.
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm, color='b')
# %%
# Data Processing Stage
# We are going to impute missing values and fix the dtypes of attributes
# so it is good to concatenating train set and test set into a dataframe and do processing on it.
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train['SalePrice'].values

all_data = pd.concat([df_train, df_test]).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape
# %%
# Printing total missing values and percentage of empytness of attributes
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count()) * 100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# %%
# Plotting missing value percentage of the attributes
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=missing_data[:20].index, y='Percent',
            data=missing_data[:20], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(title='Percent of missing values')
# %%
# The dataset description page was saying that some of the features’ missing values indicate that there is no such a thing.
# For example, If there is a house that has no pool, the dataset will indicate ‘PoolQC’ of the house as empty/missing value.
# Altough it is best to delete features if it has empty values more than %85, we avoid to delete in this problem.
# ‘PoolQC’ would be an important feature so if we delete it, we could lose some insightful information.
# Instead we had inserted ‘None’ value as indicating emptyness to the missing values.
# Morover this method was not applicable to the some features.
# Those are just missing values that does not indicate empytiness of that thing
# so we have filled missing values with mean or median value of the that particular feature

# Imputing missing values of attributes

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    all_data[col] = all_data[col].fillna(0)

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(
    all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data['Electrical'] = all_data['Electrical'].fillna(
    all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(
    all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(
    all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(
    all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(
    all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# %%
# Check remaining missing values if any
all_data.isnull().values.any()
# %%
# There were some numeric types that their dtype was not numeric such as attribute YrSold.
# We are converting them to the string types and in the next cell, encoded all categorical features with the LabelEncoder function from sklearn
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# %%
categorical_features = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
                        'YrSold', 'MoSold']

# process columns, apply LabelEncoder to categorical features
for col in categorical_features:
    label = LabelEncoder()
    all_data[col] = label.fit_transform(list(all_data[col].values))
# %%
# We are doing a little bit feature engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + \
    all_data['1stFlrSF'] + all_data['2ndFlrSF']
# %%
numerical_features = all_data.dtypes[all_data.dtypes != 'object'].index

# Checking the skewness of all numerical features
skewed_features = all_data[numerical_features].apply(
    lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_features})
skewness.head(10)
# %%
# Plotting box plot of the attributes
fig, ax = plt.subplots(figsize=(20, 15))
ax.set_xscale("log")
ax = sns.boxplot(data=all_data[numerical_features], orient='h')
# %%
# We are applying Box Cox transform on the skewed numerical attributes
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(
    skewness.shape[0]))

skewed_features = skewness.index
lam = 0.15
for feature in skewed_features:
    all_data[feature] = boxcox1p(all_data[feature], lam)
# %%
# Getting dummies
all_data = pd.get_dummies(all_data)
all_data.shape
# %%
# We had appended train set and test set before.
# Now we are seperating them
df_train = all_data[:ntrain]
df_test = all_data[ntrain:]
# %%
# MODELLING

# Just defining some variables to future use
score_calc = 'neg_mean_squared_error'
cross_val_n = 5
# %%
# Defining a function that finds best parameters of the estimator
# and returns the best score on the test set of that estimator.
def grid_overview(grid):
    best_score = np.sqrt(-grid.best_score_)
    print("Estimator: {}".format(grid.estimator))
    print("The best score is {:.4f}\n".format(best_score))
    print("The best parameters are: ")
    for key, value in grid.best_params_.items():
        print("{0}: {1}".format(key, value))
    print("\nThe best estimator is: {}".format(grid.best_estimator_))

    return best_score
# %%
# Some models requires appropriate parameters and it is exhausting to try all of them manually.
# so instead, we have defined some possible parameters for the model
# and used GridSearchCV function from sklearn to do exhaustive search over defined parameter values for the model/estimator

# Defining Linear Regression estimator and possible parameters
linear_regression = LinearRegression()
parameters = {'fit_intercept': [True, False],
              'normalize': [True, False],
              'copy_X': [True, False]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_linear = GridSearchCV(
    linear_regression, parameters, cv=cross_val_n, scoring=score_calc)

grid_linear.fit(df_train, y_train)  # fitting
linear_score = grid_overview(grid_linear)  # getting best score
# %%
# Defining Ridge Regression estimator and possible parameters
ridge = Ridge()
parameters = {'alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 1],
              'normalize': [True, False],
              'tol': [1e-06, 5e-06, 1e-05, 5e-05]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_ridge = GridSearchCV(
    ridge, parameters, cv=cross_val_n, scoring=score_calc)

grid_ridge.fit(df_train, y_train)  # fitting
ridge_score = grid_overview(grid_ridge)  # getting best score

# %%
# Defining Ridge Regression estimator and possible parameters
lasso = Lasso()
parameters = {'alpha': [1e-03, 0.01, 0.1, 0.5, 0.8, 1],
              'normalize': [True, False],
              'tol': [1e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_lasso = GridSearchCV(
    lasso, parameters, cv=cross_val_n, scoring=score_calc)

grid_lasso.fit(df_train, y_train)  # fitting
lasso_score = grid_overview(grid_lasso)  # getting best score
# %%
# Defining ElasticNet Regression estimator and possible parameters
elasticnet = ElasticNet()
parameters = {'alpha': [0.1, 1.0, 10],
              'max_iter': [1000000],
              'l1_ratio': [0.04, 0.05],
              'fit_intercept': [False, True],
              'normalize': [True, False],
              'tol': [1e-02, 1e-03, 1e-04]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_elasticnet = GridSearchCV(
    elasticnet, parameters, cv=cross_val_n, scoring=score_calc)

grid_elasticnet.fit(df_train, y_train)  # fitting
elasticnet_score = grid_overview(grid_elasticnet)  # getting best score
# %%
# Defining Decision Tree Regression estimator and possible parameters
tree = DecisionTreeRegressor()
parameters = {'max_depth': [7, 8, 9, 10], 'max_features': [11, 12, 13, 14],
              'max_leaf_nodes': [None, 12, 15, 18, 20], 'min_samples_split': [20, 25, 30],
              'presort': [False, True], 'random_state': [5]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_tree = GridSearchCV(tree, parameters, cv=cross_val_n, scoring=score_calc)

grid_tree.fit(df_train, y_train)  # fitting
tree_score = grid_overview(grid_tree)  # getting best score
# %%
# Defining Random Forest Regression estimator and possible parameters
random_forest = RandomForestRegressor()
parameters = {'min_samples_split': [3, 4, 6, 10],
              'n_estimators': [70, 100],
              'random_state': [5]}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_random_forest = GridSearchCV(
    random_forest, parameters, cv=cross_val_n, scoring=score_calc)

grid_random_forest.fit(df_train, y_train)  # fitting
random_forest_score = grid_overview(grid_random_forest)  # getting best score

# %%
# Defining KNN Regression estimator and possible parameters
knn = KNeighborsRegressor()
parameters = {'n_neighbors': [3, 4, 5, 6, 7, 10, 15],
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute']}

# Using GridSearchCV on the estimator to find best parameters of the estimator
grid_knn = GridSearchCV(knn, parameters, cv=cross_val_n,
                        scoring=score_calc, refit=True)

grid_knn.fit(df_train, y_train)  # fitting
knn_score = grid_overview(grid_knn)  # getting best score
# %%
# Gathering all scores in a list
all_scores = [linear_score, ridge_score, lasso_score, elasticnet_score,
              tree_score, random_forest_score, knn_score]

all_regressors = ['Linear', 'Ridge', 'Lasso', 'ElaNet', 'DTR', 'RF', 'KNN']
# %%
# Plotting RMSE Score of all estimators
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=all_scores, y=all_regressors, ax=ax).set(
    title='RMSE Score of all Estimators', xlabel='RMSE Score', ylabel='Estimators/Models')
# %%
# Making prediction based on defined models
pred_linear = grid_linear.predict(df_test)
pred_ridge = grid_ridge.predict(df_test)
pred_lasso = grid_lasso.predict(df_test)
pred_elanet = grid_elasticnet.predict(df_test)
pred_dtr = grid_tree.predict(df_test)
pred_rf = grid_random_forest.predict(df_test)
pred_knn = grid_knn.predict(df_test)

predictions = {'Linear': pred_linear,
               'Ridge': pred_ridge,
               'Lasso': pred_lasso,
               'ElaNet': pred_elanet,
               'DTR': pred_dtr,
               'RandomF': pred_rf,
               'KNN': pred_knn}
# %%
# calculating correlations between predictions
df_predictions = pd.DataFrame(data=predictions)
df_predictions.corr()
# %%
# plotting correlations heatmap between predictions
pred_corr = df_predictions.corr()
fig, ax = plt.subplots(figsize=(7, 7))
sns.set(font_scale=1.25)
sns.heatmap(pred_corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
            'size': 10}, yticklabels=df_predictions.columns, xticklabels=df_predictions.columns)
# %%
