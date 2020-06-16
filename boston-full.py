import pandas as pd
import scipy.stats as stats
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time

from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras import backend

from wandb.keras import WandbCallback
import wandb

sns.set()

wandb.init(project="boston-dataset")

df = pd.read_csv('housing.csv')

df.drop([353,355], inplace=True)

X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df.MEDV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def manual_r2_score(y_true, y_pred):
    sst = backend.square(y_true - backend.mean(y_true))
    ssr = backend.square(y_true - y_pred)
    return 1 - (backend.sum(ssr)/backend.sum(sst))

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
  ])

    model.compile(optimizer='adam',
                  loss='mse', metrics=[manual_r2_score, 'mae'])
    return model

model = build_model()

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

EPOCHS = 900
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, batch_size=128, 
    validation_split = 0.2, 
    verbose=2, 
    callbacks=[WandbCallback()])

score = model.predict(X_test)

print(explained_variance_score(y_test, score))
print(r2_score(y_test, score))