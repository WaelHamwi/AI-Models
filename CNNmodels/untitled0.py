# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 01:50:04 2022

@author: WAEL
"""
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

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)



X = X / 255.0 # victorization

dense_layers = [0]
layer_sizes = [32]
conv_layers = [2]

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
  
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
  
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
  
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
  
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2)
#layer_names=[layer.output for layer in model.layers[1:]]
#print(layer_names)
          
print("pactorial graaaaaaaaaaaaaaaaaaaaaaaaaaaph")    
graph=plot_model(model,to_file='my_model.png',show_shapes=True)
#model.save('32x2x0CNN.model')
model.summary()
#visualkeras.layered_view(<model>)
builder([1, 10, 10, 5, 5, 2, 1], bmode="night")
print("-------------------------------------------wvisualization my model")
visualkeras.layered_view(model, to_file='output.png').show()
print('3d visualization')
neuralplot(model=model,grid=True,connection=True,linewidth=0.1)


