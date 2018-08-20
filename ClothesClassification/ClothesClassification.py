def showPredictions(test_labels, predictions):
    for i in range (0, len(predictions)):
        print('Prediction for label ' +str(i) + ' is ' + str(np.argmax(predictions[i])) + ' should be: ' + str(test_labels[i]))



print('Importing tf and keras')
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print('Importing numpy and matplotlib')
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('Importing TestMatplotLib')
import TestMatplotLib as testplt

print(tf.__version__)

print('Importing fashion_mnist')
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
len(train_labels)

#testplt.showImg(train_images[0])
#testplt.showImages(train_images, train_labels, class_names)

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
showPredictions(test_labels, predictions)