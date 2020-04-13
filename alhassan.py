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

epoch_number = 10

train_data = np.concatenate((train_data,test_data))
train_labels = np.concatenate((train_labels,test_labels))
init = tf.keras.initializers.glorot_normal(seed=None)

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
    tf.keras.layers.Dense(2, activation='softmax',kernel_initializer=init) ])
    
model.summary()

#Compile the network
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

training = tf.reshape(train_data, [-1, 150, 150, 1])
labels = train_labels

validation = tf.reshape(validation_data, [-1,150,150,1])

history = model.fit(training,labels,validation_data=(validation,validation_labels),batch_size=50,epochs=epoch_number,verbose=1) 

#test = tf.reshape(test_data, [-1, 150, 150, 1])
#test_loss, test_acc = model.evaluate(test,test_labels, verbose=1)
#print('\nTest accuracy:', test_acc)

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

#model.save_weights('first15')

epochs_range = range(epoch_number)

#np.save('first1acc5.npy', acc)
#np.save('first1loss5.npy',loss)
#np.save('first1vallacc5.npy', val_acc)
#np.save('first1valloss5.npy', val_loss)

#prediction = model.predict_classes(validation)
#np.save('first1predict5.npy', prediction)
