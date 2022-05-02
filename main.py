#  Dopamine is a type of neurotransmitter. Your body makes it, and your nervous system uses it to send
#  messages between nerve cells.

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/faiqa/Downloads/Flask/Parkinson_Disease_Data_Science_Project/parkinsons.data")
# print(df.head())

# Get the features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Get the count of each label (0 and 1) in labels
print(labels[labels == 1].shape[0],
      labels[labels == 1].shape[0])

# Scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model
model = XGBClassifier()
model.fit(x_train, y_train)

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred) * 100)
