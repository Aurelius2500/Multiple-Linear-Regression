# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Multiple Linear Regression With Python
Spyder version 5.3.3
"""

# This tutorial builds on the Linear Regression tutorial.
# Therefore, a lot of content there will not covered here
# Instead, we will cover more statistical inference with regression, as well as some more advanced programming concepts
# Import the packages required for the code to run

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# As always, run the following lines to install the packages if you do not have them

# pip install pandas
# pip install sklearn

# This is a dataframe that has information regarding traffic accidents in Arizona
# The link is in kaggle and can be accessed in the description

accidents_df = pd.read_csv('C:/Videos/Arizona Accidents Dataset/accident.csv')

# Thie code below helps us subsetting the dataset
# We will use FATALS as the target, while the other features are predictors
# This can be interpreted as we wanting to see if we can use the other variables to predict the number of fatalities

accidents_df_sub = accidents_df[['PERSONS', 'INDIAN_RES', 'FATALS', 'A_TOD', 'A_DOW']]

# PERSONS is the number of persons involved in the accident
# INDIAN_RES tells us if the accident happened in an Indian Reservation
# FATALS is the number of fatalities
# A_TOD is the time of day
# A_DOW is the day of week

# First, let's see if there are any nulls

accidents_df_sub.isna().sum()

# We cannot use unique on the dataframe. However, we can take advantage of loops for this
# This is a while loop, that will go over the i index until it is more than the number of columns
# The len function allows us to get how many columns the dataset has
# i += 1 adds a one to the current value, and it is the same as i = i + 1

accidents_df_sub.columns
i = 0
while i < len(accidents_df_sub.columns):
    print(accidents_df_sub.columns[i] + " unique values:")
    print(accidents_df_sub.iloc[:, i].unique())
    i += 1

# This while loop allows us to get the unique values for the whole dataframe
# Define our X and y

X = accidents_df_sub.drop('FATALS', axis = 'columns')

y = accidents_df_sub['FATALS']

# Import the linear regression model from sklearn

linear_model = LinearRegression()

# Fit the model in all the variables for our original X
# We will not perform train_test_split in this video due to time constraints, it will be in a future video

linear_model.fit(X, y)

# Then get the score of the model

linear_model.score(X, y)

# Unfortunately we cannot plot this regression as easily as we did in Simple Linear Regression
# This is because instead of 
# This is due because instead of being a line, this model is a hyperplane as we are going into more dimensions
# However, we can still get some information about the model looking at the intercept and coefficients
# More advanced hypothesis testing can be performed using Statsmodels

# Find the incercept of the model
linear_model.intercept_

# And the coefficients of the model
linear_model.coef_

# This is not very informative, but with the help of the while loop, we can get the names of the predictors printed
# We can interpret the coefficients as the change in the prediction given a 1-unit change in the predictor
# Because of the expansion of the linear equation, all the other predictors must be kept constant
# This is a strong assumption of this model, that the predictors are independent
i2 = 0
print(f'Model Intercept: {linear_model.intercept_}')
while i2 < len(X.columns):
    print(X.columns[i2] + ' Coefficient')
    print(linear_model.coef_[i2])
    i2 += 1
    
# There are ways to visualize regressions with 2 predictors, but they are outside the scope of this video    
# Instead, we can fit each of the four predictors with FATALS separately

linear_model_1 = LinearRegression()
linear_model_2 = LinearRegression()
linear_model_3 = LinearRegression()
linear_model_4 = LinearRegression()

linear_model_1.fit(X.iloc[:,0].values.reshape(-1, 1), y)
linear_model_2.fit(X.iloc[:,1].values.reshape(-1, 1), y)
linear_model_3.fit(X.iloc[:,2].values.reshape(-1, 1), y)
linear_model_4.fit(X.iloc[:,3].values.reshape(-1, 1), y)

# The code below tells us the variable being fit using iloc
# f in print allows us to print a variable easily

print(f"The score for {X.columns[0]} model: ")
print(linear_model_1.score(X.iloc[:,0].values.reshape(-1, 1), y))
print(f"Model Intercept: {linear_model_1.intercept_}")
print(f"Model Coefficient: {linear_model_1.coef_}")

print(f"The score for {X.columns[1]} model: ")
print(linear_model_2.score(X.iloc[:,1].values.reshape(-1, 1), y))
print(f"Model Intercept: {linear_model_2.intercept_}")
print(f"Model Coefficient: {linear_model_2.coef_}")

print(f"The score for {X.columns[2]} model: ")
print(linear_model_3.score(X.iloc[:,2].values.reshape(-1, 1), y))
print(f"Model Intercept: {linear_model_3.intercept_}")
print(f"Model Coefficient: {linear_model_3.coef_}")

print(f"The score for {X.columns[3]} model: ")
print(linear_model_4.score(X.iloc[:,3].values.reshape(-1, 1), y))
print(f"Model Intercept: {linear_model_4.intercept_}")
print(f"Model Coefficient: {linear_model_4.coef_}")

# Predict with the models

first_predictions = linear_model_1.predict(X.iloc[:,0].values.reshape(-1, 1))
second_predictions = linear_model_2.predict(X.iloc[:,1].values.reshape(-1, 1))
third_predictions = linear_model_3.predict(X.iloc[:,2].values.reshape(-1, 1))
fourth_predictions = linear_model_4.predict(X.iloc[:,3].values.reshape(-1, 1))
multi_predictions = linear_model.predict(X)

# Plot the results of the first model

plt.ticklabel_format(style='plain')
plt.xlabel("Persons")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Persons vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['PERSONS'], y, color = "Blue", s = 10)
plt.plot(X['PERSONS'], first_predictions, color = "black", linewidth = 2)
plt.show()

# Plot the results of the second model

plt.ticklabel_format(style='plain')
plt.xlabel("Indian Reservation")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Indian Reservation vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['INDIAN_RES'], y, color = "Blue", s = 10)
plt.plot(X['INDIAN_RES'], second_predictions, color = "black", linewidth = 2)
plt.show()

# Plot the results of the third model

plt.ticklabel_format(style='plain')
plt.xlabel("Time of Day")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Time of Day vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['A_TOD'], y, color = "Blue", s = 10)
plt.plot(X['A_TOD'], third_predictions, color = "black", linewidth = 2)
plt.show()

# Plot the results of the fourth model

plt.ticklabel_format(style='plain')
plt.xlabel("Day of Week")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Day of Week vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['A_DOW'], y, color = "Blue", s = 10)
plt.plot(X['A_DOW'], fourth_predictions, color = "black", linewidth = 2)
plt.show()

# We can compare the regression line of the full model with the simpler Persons model
# See how the multiple linear regression does not seem to be linear in two dimensions

plt.ticklabel_format(style='plain')
plt.xlabel("Persons")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Persons vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['PERSONS'], y, color = "Blue", s = 10)
plt.plot(X['PERSONS'], first_predictions, color = "black", linewidth = 1)
plt.plot(X['PERSONS'], multi_predictions, color = "orange", linewidth = 0.1)
plt.show()

# We can make this graph cleaner by using the data poits instead of the regression line

plt.ticklabel_format(style='plain')
plt.xlabel("Persons")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Persons vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X['PERSONS'], y, color = "Blue", s = 10)
plt.plot(X['PERSONS'], first_predictions, color = "black", linewidth = 0.5)
plt.scatter(X['PERSONS'], multi_predictions, color = "orange", s = 2)
plt.show()

# As we have compared multiple models, compare the two variables that have the best score together, PERSONS and INDIAN_RES

linear_model_two_var = LinearRegression()

X_selected = X[['PERSONS', 'INDIAN_RES']]

linear_model_two_var.fit(X_selected, y)

two_var_predictions = linear_model_two_var.predict(X_selected)

# Plot the results of the last model

plt.ticklabel_format(style='plain')
plt.xlabel("Persons and Indian Reservation")
plt.ylabel("Fatalities")
plt.title("Linear Regression on Persons and Indian Reservation vs Fatalities")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X["PERSONS"], y, color = "Blue", s = 10)
plt.scatter(X["PERSONS"], two_var_predictions, color = "black", s = 10)
plt.show()

# If we were to graph this with the indian reservation indicator, then the plane would be more visible

# Finally, we can get the predictions on 
# This corresponds to 17 persons involved, in an indian reservation, at night, and on a weekend
new_obs = np.array([17, 1, 2, 2])

linear_model.predict(new_obs.reshape(1, -1))

manual_prediction = linear_model.intercept_ + linear_model.coef_[0] * new_obs[0] + linear_model.coef_[1] * new_obs[1] + linear_model.coef_[2] * new_obs[2] + linear_model.coef_[3] * new_obs[3]

# Compare this result to the first model

linear_model_1.predict(new_obs[0].reshape(1, -1))

# Sometimes a simpler model is better for the data that you are working on
# Even if linear regression does not really fit this data, it is a good exploratory data analysis tool
# Regression models are also one of the easiest and more interpretable models, being a good initial tool too to evaluate your data