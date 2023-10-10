import numpy as np

from network import Network
from fc_layer import FCLayer
from fl_layer import FlattenLayer
from sm_layer import SoftmaxLayer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_prime, relu, relu_prime
from losses import mse, mse_prime, crossEntropy, crossEntropy_prime

from keras.datasets import cifar10
from keras.utils import to_categorical

import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# load the Cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 32*32*3)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 32*32*3)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Network
net = Network()
net.add(FCLayer(32*32*3, 128))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(128, 10))
net.add(SoftmaxLayer(10))
# Set Loss function
net.use(mse, mse_prime)
# Loads weights
net.load_weights('weights_cifar10.npy')
# Training function. Options are fit which requires a sample size or batch_fit which requires a batch size
net.batch_fit(x_train, y_train, epochs=40, learning_rate=.1, batch_size=128)
# Saves weights
net.save_weights('weights_cifar10.npy')
# set amount of test samples
samples = 6
# test those test samples
out = net.predict(x_test[:samples])
# print wether predictions are correct and the confidence percent of the predictions

names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
         4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
for i in range(samples):
    first_image = x_test[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((32, 32, 3))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    # Prediction
    prediction = np.argmax(out[i])
    confidence = np.max(out[i])
    true_value = np.argmax(y_test[i])

    print("Sample", i+1)
    if (prediction == true_value):
        print("Correct")
    else:
        print("Wrong")
    print("Guessed value:", names[prediction])
    print("Confidence:", confidence * 100, "%")
    print("Correct value:", names[true_value])
    print()
