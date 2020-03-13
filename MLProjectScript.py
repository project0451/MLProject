# Python script for ML course project
# import and clean data set on mammographic masses and run several ML models

import numpy as np
import pandas as pd
import scipy
from sklearn import model_selection
from sklearn import tree
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Read in data from file
# Treat "?" values as missing data / NaN
# use data type Int64 (capital I) to use pandas nullable integer data type
df = pd.read_csv(
    "mammographic_masses.data.txt",
    names=["BI-RADS Assessment", "Age", "Shape", "Margin", "Density", "Severity"],
    dtype="Int64", na_values="?")
# pd.Int64Dtype(),
# We don't actually need the BI-RADS data, and training the model on it would be cheating, so drop it
df.drop(axis=1, labels="BI-RADS Assessment", inplace=True)

# Create pd.NA by hand because python insists it doesn't exist when I refer to it directly
# May be related to changing "experimental" behavior of pd.NA and pandas nullable Int64
# Currently appears to be using np.nan, which pd.NA was suppoedly made specifically to replace
# This way should (hopefully) be relatively future-proof
pdNA = pd.Series([None], dtype='Int64')[0]

# Check for invalid values, and treat as missing data (NaN).  Valid values should be in these ranges
# BI-RADS: 1-5 (ordinal)
# Age: 0-100 (maybe narrower) (integer)
# Shape: 1-4 (nominal)
# Margin: 1-5 (nominal)
# Density: 1-4 (ordinal)
# Severity: 0-1 (binary class, prediction target)

df = df.where(
    cond=df.isin({
        "Age": [i for i in range(0, 121)],
        "Shape": [1, 2, 3, 4],
        "Margin": [1, 2, 3, 4, 5],
        "Density": [1, 2, 3, 4],
        "Severity": [0,1]
        }),
    other=pdNA
)

# Clean up missing valuess
# For now, simply drop any row with missing values since less than 20% of rows have missing values
# Keep a copy of the old DataFrame for XGBoost and other methods that can handle missing data
df2 = df.dropna()

# Convert to standard int32 now that we don't need nullable data types anymore so that sklearn doesn't choke on pandas' nullable Int64 type
df2 = df2.astype('int32')

# Later, only drop rows with missing age (very few) or missing severity (useless), and try a more sophisticated approach with for the others
# df.drop(axis=0, subset=['Age', 'Severity'])

# XGBoost doesn't like pandas nullable Int64 either, so convert the NaNs to -1 and convert to int32
df_xgb = df.mask(df.isna(), other=-1).astype('int32')


# Extract data and target values as numpy arrays
data = df2.drop(axis=1, labels=["Severity"]).to_numpy()
target = df2['Severity'].to_numpy()

# Separate data into test and multiple training sets for K-fold validation
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.4)

# First machine learning model: Decision Tree classifier
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(x_train, y_train)

# Report accuracy
print("Accuracy: %s" % tree_clf.score(x_test, y_test))

# K-fold validation
scores = model_selection.cross_val_score(tree_clf, data, target, cv=5)
print("Validation: %s" % scores)
print(scores)
print("Mean: %s" % scores.mean())

# Random forest using XGBoost
# Split into training and test data and convert into XGBoost DMatrix containers
xgb_x_train, xgb_x_test, xgb_y_train, xgb_y_test = model_selection.train_test_split(
    df_xgb.drop(axis=1, labels='Severity'),
    df_xgb['Severity'],
    test_size=0.4)
xgb_train = xgb.DMatrix(xgb_x_train, label=xgb_y_train, missing=-1)
xgb_test = xgb.DMatrix(xgb_x_test, label=xgb_y_test, missing=-1)

# Define XGBoost model hyperparameters
param = {
    'max_depth': 10,
    'eta': 0.25,
    'objective': "binary:hinge",
}
epochs = 10

# Train XGBoost model on the data
xgb_model = xgb.train(param, xgb_train, epochs)

# Evaluate model
xgb_predictions = xgb_model.predict(xgb_test)
print("XGBoost model accuracy: %s" % accuracy_score(xgb_y_test, xgb_predictions))

# Use model_selection.GridSearchCV to find optimal hyperparametr values
# General syntax is:
#   GridSearchCV(estimator, param_grid, scoring, n_jobs, cv, pre_dispatch)
# where
#   estimator = the model to apply GridSearchCV to
#   param_grid = dictionary of hyperparamater names and lists of values
#       {'param_1': [list of values], 'param_2': [list of values], etc.}
#       will try every combination of values for every hyperparameter
#   scoring = string or callable specifying scoring method to use
#       uses estimator.score( ) by default if not specified
#       try using 'accuracy' use metrics.accuracy_score for a classifier
#   n_jobs = maximum number of parallel jobs, 1 if not specified
#       -1 for all available processors, -n for all but (n-1) processors
#   cv = number of cross-validation folds, uses 5-fold if not spedified
#   pre_dispatch = limit number of copies of data in memory if n_jobs !=1, default is '2*n_jobs'
#
# Also need the scikit-learn wrapper class for XGBoost
# xgb.XGBClassifier(objective='binary:logistic',)
# xgb.XGBRFClassifier(learning_rate=1,)
# For this case, we want objetive='binary:hinge'
# Relevant parameters
#   n_estimators = number of trees in random forest to fit(RF only)
#   max_depth = maximum tree depth for base learners
#   learning_rate = (float) "eta" in xgb, boosted learning rate
#   objective = learning task and objective, or custom objective function
#   booster = 'gbtree', 'gblinear', or 'dart'
#   tree_method = ??? (leave as default for now)
#   n_threads = number of threads to use
#   gamma = (float) minimum loss reduction requared to make a furhter partition
#   min_child_seight = minimum sum of instance weight needed in a child
#   missing = value to treat as missing
#   num_parallel_tree = used for random forest

xgb_rf_model = xgb.XGBRFClassifier(objective="binary:hinge", missing=-1)
xgb_param_grid = {
    "n_estimators": [50,100,200],
    "max_depth": [4,6,8,10,12],
    "learning_rate": [0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2],
    "num_parallel_tree": [50, 100, 200],
    "gamma": [0,0.1,0.25],
    "min_child_weight": [0.5,1,2]
}

xgb_grid_rf_clf = model_selection.GridSearchCV(xgb_rf_model, xgb_param_grid, n_jobs=4)
#xgb_grid_rf_clf = xgb_rf_model
xgb_grid_rf_clf.fit(xgb_x_train, xgb_y_train)

print("")
print(xgb_grid_rf_clf.best_score_)
print("")
print(xgb_grid_rf_clf.best_params_)

print("Accuracy: %s" % xgb_grid_rf_clf.score(xgb_x_test, xgb_y_test))
scores = model_selection.cross_val_score(xgb_grid_rf_clf, df_xgb.drop(axis=1, labels=['Severity']), df_xgb['Severity'])
print("Cross Validation: %s" % scores)
print("Mean: %s" % scores.mean())
print(xgb_grid_rf_clf.get_params())