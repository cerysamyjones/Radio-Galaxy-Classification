# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:30:16 2020

@author: Cerys
"""
import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#FIRST
#train_data = np.load('firsttrainingdata.npy')
#train_labels = np.load('firsttraininglabels.npy')
#test_data = np.load('firsttestdata.npy')
#test_labels = np.load('firsttestlabels.npy')
#validation_data = np.load('firstvalidation.npy')
#validation_labels = np.load('firstvalidationlabels.npy')

#VLASS
train_data = np.load('trainingdata.npy')
train_labels = np.load('traininglabels.npy')
test_data = np.load('testdata.npy')
test_labels = np.load('testlabels.npy')
validation_data = np.load('validation.npy')
validation_labels = np.load('validation_labels.npy')

model = 'first15' 
epoch_number = 700
init = tf.keras.initializers.GlorotNormal(seed=None)

# Alhassan  Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1),kernel_initializer=init),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3),activation='relu',kernel_initializer=init),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(194, (3,3), activation='relu',kernel_initializer=init),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(194, activation='relu',kernel_initializer=init),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax') ])
    
model.summary()

#for layer in model.layers[:5]:
#    layer.trainable=False

#Compile the network
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

training = tf.reshape(train_data, [-1, 150, 150, 1])
labels = train_labels

validation = tf.reshape(validation_data, [-1,150,150,1])

history = model.fit(training,labels,validation_data=(validation,validation_labels),batch_size=128,epochs=epoch_number,verbose=1) 

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs_range = range(epoch_number)

np.save('transferfirstvlass1valacc3.npy',val_acc)
np.save('transferfirstvlass1valloss3.npy',val_loss)

prediction = model.predict_classes(validation)
np.save('transferfirstvlass1predict3.npy',prediction)

