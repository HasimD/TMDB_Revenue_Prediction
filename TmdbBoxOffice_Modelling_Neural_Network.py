# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

train = pd.read_csv("train_features.csv") 
test = pd.read_csv("test_features.csv")
train=train.replace([np.inf,-np.inf],np.nan)
test=test.replace([np.inf,-np.inf],np.nan)
train.fillna(value=0.0, inplace = True)
test.fillna(value=0.0, inplace = True)
train['revenue'] = np.log1p(train['revenue'])
y_train = train['revenue'].values
train.drop(['revenue'], axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.transform(test)
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 522))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))


model.compile(optimizer = "rmsprop", loss = 'mean_squared_error',metrics =["mse"])

model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)


sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = np.expm1(y_pred)
df_sub=df_sub.replace([np.inf,-np.inf],0.0)
print(df_sub['revenue'])
df_sub.to_csv("NN_submission.csv", index=False)