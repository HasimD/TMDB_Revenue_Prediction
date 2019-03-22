import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

trainData = pd.read_csv("train_features.csv")
testData = pd.read_csv("test_features.csv")
trainData = trainData.replace([np.inf, -np.inf], np.nan)
testData = testData.replace([np.inf, -np.inf], np.nan)
trainData.fillna(value=0.0, inplace=True)
testData.fillna(value=0.0, inplace=True)

trainResult = trainData[['revenue']]
trainData.drop('revenue', axis=1, inplace=True)

parameters = {'n_jobs': (1, 25, 50, 100, 250, 500, 750, 1000)}
reg = LinearRegression()
reg = GridSearchCV(reg, parameters, cv=2)
reg.fit(trainData, trainResult)

prediction = reg.predict(testData)

score = reg.score(trainData, trainResult)
print(score)
dataframe = pd.DataFrame(data=prediction.flatten())
dataframe.columns = ['revenue']

sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = dataframe['revenue']
df_sub.to_csv("burak_submission.csv", index=False)

trainResult = trainData.iloc[:, -1]
trainData.drop('revenue', axis=1, inplace=True)


""" LINEAR REGRESSION """
parameters = {'n_jobs': (1, 25, 50, 100, 250, 500, 750, 1000)}
reg = LinearRegression()
reg = GridSearchCV(reg, parameters, cv=2)
reg.fit(trainData, trainResult)
prediction = reg.predict(testData)

sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = sub['id']
df_sub.to_csv("xgboost_submission.csv", index=False)


dataframe = pd.DataFrame(data=prediction.flatten())
dataframe.columns = ['revenue']

sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = dataframe['revenue']
df_sub.to_csv("burak_submission.csv", index=False)

"""
// SUPPORT VECTORE MACHINE
svc = SVR()
svc = GridSearchCV(svc, parameters, cv=2)
svc.fit(trainData, trainResult)
prediction = svc.predict(trainData)
dataframe = pd.DataFrame(data=prediction.flatten())
dataframe.columns = ['revenue']

sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = dataframe['revenue']
df_sub.to_csv("burak_submission.csv", index=False)
"""