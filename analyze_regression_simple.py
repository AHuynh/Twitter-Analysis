import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 0:-2].values
y = dataset.iloc[:, -2].values
########################################################################################
## Data preprocesssing. Encoding categorical data.

#print(X[0:3])
# [0.0 60 2023 9 13 21 2 0 False 1 0 0 0]

## Time. Use One-Hot Encoding, as year/month/date are categorical. Though there is some relation
# due to ordering, I want to avoid the issue of rollover times (eg. 31st -> 1st, Dec -> Jan) not
# being captured by ordinal encoding. I could use a sine/cosine transformation, but for this first
# pass I'll keep it simple.
# Year: [2013-2024). Will end up with 10 columns (11 years - 1, drop to avoid multicollinearity)
# Month: [0-13). End up with 11 columns.
# Day: [0-31). End up with 30 columns.
# Hour: [0-24). End up with 23 columns.
# Day of week: [0-8). End up with 7 columns.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3, 4, 5])], remainder='passthrough', sparse_threshold=0)
X = np.array(ct.fit_transform(X))

#print(X[0:3])

########################################################################################
## Data set splitting. Split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

########################################################################################
## Feature scaling. Need to scale for some regressions. Scale features only.
sc = StandardScaler()
X_train_sc = X_train
X_test_sc = X_test
X_train_sc[:, -6:] = sc.fit_transform(X_train_sc[:, -6:])
X_test_sc[:, -6:] = sc.transform(X_test_sc[:, -6:])

#print(X_train[0:3])
#print('OK!')

########################################################################################
## Multiple Linear Regression
mlr_regressor = LinearRegression()
mlr_regressor.fit(X_train_sc, y_train)
y_pred_mlr = mlr_regressor.predict(X_test_sc)

## MSE accuracy measurement
print('MSE (MLR):', mean_squared_error(y_test, y_pred_mlr))

########################################################################################
## LDA dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_sc, y_train)
X_test_lda = lda.transform(X_test_sc)

## Multiple Linear Regression (LDA)
mlr_lda_regressor = LinearRegression()
mlr_lda_regressor.fit(X_train_lda, y_train)
y_pred_mlr_lda = mlr_lda_regressor.predict(X_test_lda)

## MSE accuracy measurement(LDA)
print('MSE (MLR+LDA):', mean_squared_error(y_test, y_pred_mlr_lda))
print(' Score (Train):', mlr_lda_regressor.score(X_train_lda, y_train))
print(' Score (Test):', mlr_lda_regressor.score(X_test_lda, y_test))

#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


########################################################################################
## Decision Tree Regression
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)

## MSE accuracy measurement(LDA)
print('MSE (DT):', mean_squared_error(y_test, y_pred_tree))
print(' Score (Train):', tree_regressor.score(X_train, y_train))
print(' Score (Test):', tree_regressor.score(X_test, y_test))


