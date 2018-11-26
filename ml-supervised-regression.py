import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

#load the new marketing data file
dataset = pd.read_csv('marketing.csv')

#create a dataframe with the dataset
df = pd.DataFrame (dataset, columns=['Demography','Marketing Spend','Revenue'])

#create X (independent variable) as a slice of the DataFrame with both required columns in it.
X= df[['Demography','Marketing Spend']]

#create Y (dependent variable) as a slice of the DataFrame with the required column in it.
Y= df[['Revenue']]

#split the provided data into test and train data.
train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.8)

#initialize LinerRegression function
regressor = lm.LinearRegression()

#fit the data
regressor.fit(train_X,train_Y)

#print the coeffecients
print (regressor.coef_)
print (regressor.intercept_)