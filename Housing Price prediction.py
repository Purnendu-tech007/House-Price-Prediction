import pandas as pd
df = pd.read_csv ('bangalore house price.csv')
df
list (df)
df.shape

# Seprating X and Y
X = df.drop('price',axis=1)
X = df.iloc [:, 0:108]
list (X)
Y = df ['price']
df.dtypes
X.shape

# Scatter Plot
import matplotlib.pyplot as plt
plt.show()

df.plot.scatter(x ='bath',y= 'price') --> No Relatiopnship
df.plot.scatter(x='balcony', y='price') --> No Relationship
df.plot.scatter(x='total_sqft_int', y='price') --> Positive
df.plot.scatter(x='bhk', y= 'price') --> No Relationship
df.plot.scatter(x='price_per_sqft', y='price') --> Positive
df.corr()

# Boxplot
df.boxplot('bath')
df.boxplot('balcony')
df.boxplot('total_sqft_int')
df.boxplot('bhk')
df.boxplot('price_per_sqft')

# Spliting the Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y)
X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

# Linear Regression Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(X_train,Y_train)
LR.intercept_
LR.coef

# Prediction
Y_pred = LR.predict(X_test)
Y_pred
Y_pred.shape

# Mean Square Error
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(Y_test,Y_pred)
MSE
r2 = r2_score (Y_test,Y_pred)
r2
print ("mean squared error:", MSE.round(2))

# if r2 value is 1 then it's good model and if 0 then its bad model
=============================================================
# Standardization

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
Xscale = sc.transform(X)
Xscale
=============================================================
# Support Vector Machine (SVM), # SVR ----> Regressor

from sklearn.svm import SVR
clf = SVR()
SVR?
clf = SVR(C=1.0, kernel='linear')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

# Mean Square Error
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(Y_test,Y_pred)
MSE
r2 = r2_score (Y_test,Y_pred)
r2
print ("mean squared error:", MSE.round(2))

==================================================================
# K_Nearest Neighbourhood (KNN)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit (X_train, Y_train)
knn

# Prediction

Y_pred = knn.predict(X_test) # Predict Results
Y_test # Actual Result

=======================================================
# Regularization
# Ridge Regression and alpha - 1.0
from sklearn.linear_model import Ridge
RR = Ridge(alpha=1.0)
RR.fit(X_train, Y_train)
RR.coef_
RR.coef_[0]

RR?

# Prediction
Y_pred_train = RR.predict(X_train)n) --------> Predicted Result
Y_train -------------------> Actual Result

Y_pred_test = RR.predict(X_test)
Y_test
# alpha = 0.5
from sklearn.linear_model import Ridge
RR = Ridge(alpha=0.5)
RR.fit(X_train, Y_train)
RR.coef_

Y_pred_train1 = RR.predict(X_train)
Y_train

# alpha = 1.5
from sklearn.linear_model import Ridge
RR = Ridge(alpha=1.5)
RR.fit(X_train, Y_train)
RR.coef_

Y_pred_train2 = RR.predict(X_train)
Y_train


import matplotlib.pylab as plt
plt.plt(range(0,108), RR.coef_[0])
====================================================================
# Lasso Regression
from sklearn.linear_model import Lasso
lss = Lasso()
lss.fit(X_train, Y_train)
lss.coef_
Y_pred = LR.predict(X_train)
MSE = mean_squared_error(Y_train, Y_pred)
print(MSE.round(2))




