# -*- coding: utf-8 -*-

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation , Flatten
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
model.add(Dense(128, kernel_initializer='normal',input_dim = 522, activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(units=1, kernel_initializer='normal',activation='linear'))
model.compile(loss=root_mean_squared_error, optimizer='adam')
model.summary()
checkpoint_name = 'Weights.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
history=model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
wights_file = 'Weights.hdf5' # choose the best checkpoint 
model.load_weights(wights_file) # load it
model.compile(loss=root_mean_squared_error, optimizer='sgd')
print('Loss:    ', history.history['loss'][-1], '\nVal_loss: ', history.history['val_loss'][-1])
score_rmse_train = model.evaluate(X_train, y_train)
print('Train Score:', score_rmse_train)
y_pred = model.predict(X_test)
sub = pd.read_csv('sample_submission.csv')
df_sub = pd.DataFrame()
df_sub['id'] = sub['id']
df_sub['revenue'] = np.expm1(y_pred)
df_sub=df_sub.replace([np.inf,-np.inf],0.0)
print(df_sub['revenue'])
df_sub.to_csv("NN_submission.csv", index=False)

