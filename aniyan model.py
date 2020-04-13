# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:30:16 2020

@author: Cerys
"""
import tensorflow as tf
import numpy as np
from astropy import stats

def train_test(data,percentage):
    '''
    Combines data sets in one 3D array, with a different label for each data set.
    Then randomly shuffles the data and splits into training and test sets.
    data = list 3D arrays containing desired data sets
    per = fraction of data to be in training set
    returns: train and test data (each a tuple containing the data and corresponding labels)
    '''
    d = np.concatenate(data,axis=0)
    n_images = len(d)
    labels = np.empty(n_images)
    i = 0
    for n in range(len(data)):
        labels[i:i+len(data[n])] = n
        i = len(data[n])
    rand_ind = np.random.permutation(range(n_images))
    d, labels = d[rand_ind], labels[rand_ind]
    n_train = np.int(np.round(n_images*percentage))
    train = (d[:n_train], labels[:n_train])
    test = (d[n_train:], labels[n_train:])
    return train, test

def sigma_clip(data,sigma):
    """
    Takes a 3D array, returns a 3D array of sigma clipped images
    """
    for i in range(len(data)):
        d = data[i,:,:]
        d[np.isnan(d)] = 0
        _,median,std = stats.sigma_clipped_stats(d, sigma=sigma)
        d[d<median+sigma*std] = median+sigma*std
        data[i,:,:] = d
        return data

fr1_augment = np.load('fr1_augment.npy')
fr2_augment = np.load('fr2_augment.npy')

fr1_augment = sigma_clip(fr1_augment,3)
fr2_augment = sigma_clip(fr2_augment,3)

train,test = train_test((fr1_augment,fr2_augment),0.8)

# Aniyan Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (10, 10), activation='relu', input_shape=(150, 150, 1),
    strides=(5, 5)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(256, (5, 5),activation='relu'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3),strides=(2, 2)),
    tf.keras.layers.Conv2D(384, (3, 3),activation='relu',strides=(1, 1)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Conv2D(384, (2, 2),activation='relu',strides=(1, 1)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Conv2D(256, (2, 2), activation='relu',padding='same'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2),padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096*2, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax') ])
    
model.summary()

# Compile the network
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training = tf.reshape(train[0], [-1, 150, 150, 1])
labels = train[1]

model.fit(training, labels,epochs=10,verbose=1) 

test = tf.reshape(test[0], [-1, 150, 150, 1])
test_labels = test[1]
test_loss, test_acc = model.evaluate(test,test_labels, verbose=1)
print('\nTest accuracy:', test_acc)

validation = np.load('validation.npy')

validation = tf.reshape(validation,[-1,150,150,1])
classes = model.predict(validation)
np.save('predicted_classes.npy',classes)