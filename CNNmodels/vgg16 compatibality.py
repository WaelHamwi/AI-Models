# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:35:45 2022

@author: WAEL
"""
from glob import glob
from keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.callbacks import TensorBoard
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)



X = X / 255.0
model = Sequential()

model.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax")) 


img1_shape = (224,224,3)
model_224 = VGG16(include_top=False, weights='imagenet', input_shape=img1_shape)
#print(model_224.summary())
vgg_model = VGG16(include_top=False, weights='imagenet')
#print(vgg_model.summary())
vgg_config = vgg_model.get_config()
h, w, c = 500, 500, 1
vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)
print(vgg_config)
vgg_updated = Model.from_config(vgg_config)
print(vgg_updated.summary())
orig_model_conv1_block1_wts = vgg_model.layers[1].get_weights()[0]

print(orig_model_conv1_block1_wts[:,:,0,0])
print(orig_model_conv1_block1_wts[:,:,1,0])
print(orig_model_conv1_block1_wts[:,:,2,0])
new_model_conv1_block1_wts = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts[:,:,0,0])
def avg_wts(weights):  
  average_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)
  return(average_weights)
vgg_updated_config = vgg_updated.get_config()
vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in range(len(vgg_updated_config['layers']))]
first_conv_name = vgg_updated_layer_names[1]
for layer in vgg_model.layers:
    if layer.name in vgg_updated_layer_names:
        if layer.get_weights() != []:
            target_layer = vgg_updated.get_layer(layer.name)
            if layer.name in first_conv_name:
                weights = layer.get_weights()[0]
                biases  = layer.get_weights()[1]
                weights_single_channel = avg_wts(weights)
                target_layer.set_weights([weights_single_channel, biases])
                target_layer.trainable = False
            else:
                target_layer.set_weights(layer.get_weights())
                target_layer.trainable = False
new_model_conv1_block1_wts_updated = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts_updated[:,:,0,0])
vgg_updated.summary()
print("----------------------------------------------------")
folders = glob(r'c:\\train')

x = Flatten()(vgg_updated.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)#for giving our model output depending on our dataset catagories

# create a model object
model = Model(inputs=vgg_updated.input, outputs=prediction)

# view the structure of the model
model.summary()
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
r=model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2, callbacks=['tensorboard'])
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
model.save('facefeatures_new_model.h5')
