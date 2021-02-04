import pickle
import numpy as np
import time as time
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras.utils import to_categorical

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

testdict = unpickle('./cifar-10-batches-py/test_batch')
datadict1 = unpickle('./cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('./cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('./cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('./cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('./cifar-10-batches-py/data_batch_5')
labeldict = unpickle('./cifar-10-batches-py/batches.meta')

X1 = datadict1['data']
X2 = datadict2['data']
X3 = datadict3['data']
X4 = datadict4['data']
X5 = datadict5['data']
Y1 = datadict1['labels']
Y2 = datadict2['labels']
Y3 = datadict3['labels']
Y4 = datadict4['labels']
Y5 = datadict5['labels']

X1 = X1.reshape(10000, 3, 32, 32).astype("int")
X2 = X2.reshape(10000, 3, 32, 32).astype("int")
X3 = X3.reshape(10000, 3, 32, 32).astype("int")
X4 = X4.reshape(10000, 3, 32, 32).astype("int")
X5 = X5.reshape(10000, 3, 32, 32).astype("int")

testDataArray = testdict["data"]
testLabelArray = testdict["labels"]

testDataArray = testDataArray.reshape(10000, 3, 32, 32).astype("int")

dataArray = np.concatenate([X1, X2])
dataArray = np.concatenate([dataArray, X3])
dataArray = np.concatenate([dataArray, X4])
dataArray = np.concatenate([dataArray, X5])

printDataArray = dataArray.transpose(0, 2, 3, 1)

labelArray = np.concatenate([Y1, Y2])
labelArray = np.concatenate([labelArray, Y3])
labelArray = np.concatenate([labelArray, Y4])
labelArray = np.concatenate([labelArray, Y5])

dataArray = dataArray.transpose(0, 2, 3, 1)
testDataArray = testDataArray.transpose(0, 2, 3, 1)

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
labelNamesArray = labeldict["label_names"]

testDataArray = np.array(testDataArray)
testLabelArray = np.array(testLabelArray)
testLabelArray = to_categorical(testLabelArray)
dataArray = np.array(dataArray)
labelArray = np.array(labelArray)
labelArray = to_categorical(labelArray)
print('labels: ', labelArray.shape)

print(dataArray.shape)
print(testDataArray.shape)
print(testLabelArray.shape)

inputs = keras.Input(shape=(32, 32, 3), name='img')

x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(inputs)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.1)(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='sigmoid')(x)

myCallbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        mode='auto',
        restore_best_weights=False
        )
]

model = keras.Model(inputs=inputs, outputs=outputs, name='model')
print(model.summary())

model.compile(
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Nadam(0.00005),
    metrics = ['accuracy']
)

history = model.fit(dataArray, labelArray, batch_size = 10, epochs = 100, validation_split = 0.1, callbacks=[myCallbacks])

test_scores = model.evaluate(testDataArray, testLabelArray, verbose = 2)
print('Test loss: ', test_scores[0])
print('Test accuracy: ', test_scores[1])