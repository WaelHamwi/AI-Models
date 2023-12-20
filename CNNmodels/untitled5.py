# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:25:38 2022

@author: WAEL
"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from keras.utils.vis_utils import plot_model
#from keras.utils.vis_utils_util import plot_model
import visualkeras
from eiffel2 import builder
import neuralplot as neuralplot
import tensorflow as tf
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScale

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)



X = X / 255.0 # victorization

dense_layers = [0]
layer_sizes = [32]
conv_layers = [2]


def build_cnn(activation = 'relu',
              dropout_rate = 0.2,
              optimizer = 'Adam'):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation=activation,input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(
            loss='binary_crossentropy', 
            optimizer=optimizer, 
            metrics=['accuracy']
        )
    
    return model
param_grid = {
              'epochs':[1,2,3],
              'batch_size':[128]
              #'epochs' :              [100,150,200],
              #'batch_size' :          [32, 128],
              #'optimizer' :           ['Adam', 'Nadam'],
              #'dropout_rate' :        [0.2, 0.3],
              #'activation' :          ['relu', 'elu']
             }

model = KerasClassifier(build_fn = build_cnn, verbose=0)
  

params={'batch_size':[100, 20, 50, 25, 32],
		'nb_epoch':[200, 100, 300, 400],
		'unit':[5,6, 10, 11, 12, 15],
		
		}
gs=GridSearchCV(estimator=model, param_grid=params, cv=10)
# now fit the dataset to the GridSearchCV object.
gs = gs.fit(X, y)
best_params=gs.best_params_
accuracy=gs.best_score_
print(best_params)
print(accuracy)
model = model(X,y, model, 
                                        param_grid, cv=5, scoring_fit='neg_log_loss')

print(model.best_score_)
print(model.best_params_)