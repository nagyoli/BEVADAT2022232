#!/usr/bin/env python
# coding: utf-8

# In[45]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras.engine.sequential
import numpy as np
from typing import Tuple

'''
Készíts egy metódust ami a cifar100 adatbázisból betölti a train és test adatokat. (tf.keras.datasets.cifar100.load_data())
Majd a tanitó, és tesztelő adatokat normalizálja, és vissza is tér velük.


Egy példa a kimenetre: train_images, train_labels, test_images, test_labels
függvény neve: cifar100_data
'''
def cifar100_data() -> Tuple:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels


# In[46]:


'''
Készíts egy konvolúciós neurális hálót, ami képes felismerni a képen mi van a 100 osztály közül.
A háló kimenete legyen 100 elemű, és a softmax aktivációs függvényt használja.
Hálon belül tetszőleges számú réteg lehet..


Egy példa a kimenetre: model,
return type: keras.engine.sequential.Sequential
függvény neve: cifar100_model
'''
def cifar100_model() -> keras.engine.sequential.Sequential:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=tf.keras.activations.swish, input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.activations.swish))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.activations.swish))
    model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.activations.swish))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation=tf.keras.activations.swish))
    model.add(layers.Dense(100, activation='softmax'))

    return model


# In[47]:


'''
Készíts egy metódust, ami a bemeneti hálot compile-olja.
Optimizer: Adam
Loss: SparseCategoricalCrossentropy(from_logits=False)

Egy példa a bemenetre: model
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_compile
'''
def model_compile(model: keras.engine.sequential.Sequential) -> keras.engine.sequential.Sequential:
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model


# In[48]:


'''
Készíts egy metódust, ami a bemeneti hálót feltanítja.

Egy példa a bemenetre: model,epochs, train_images, train_labelsz
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_fit
'''
def model_fit(model: keras.engine.sequential.Sequential, epochs: int, train_images, train_labels) -> keras.engine.sequential.Sequential:
    model.fit(train_images, train_labels, epochs=epochs, verbose=1)

    return model


# In[49]:


'''
Készíts egy metódust, ami a bemeneti hálót kiértékeli a teszt adatokon.

Egy példa a bemenetre: model, test_images, test_labels
Egy példa a kimenetre: test_loss, test_acc
return type: float, float
függvény neve: model_evaluate
'''
def model_evaluate(model, test_images, test_labels) -> Tuple[float, float]:
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

    return test_loss, test_acc

