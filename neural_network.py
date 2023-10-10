import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [
                   [1, 1]], [[1, 1]], [[0, 1]], [[1, 0]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]], [[0]], [[1]], [[1]]])

# network
net = Network()
net.add(FCLayer(2, 10))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# Loss function options
net.use(mse, mse_prime)
# Load weights
net.load_weights('weights_andor.npy')
# Training function. Options are fit which requires a sample size or batch_fit which requires a batch size
net.batch_fit(x_train, y_train, epochs=10000, learning_rate=.1, batch_size=1)
# Save weights
net.save_weights('weights_andor.npy')
# test
out = net.predict([[[0, 0]], [[1, 1]], [[1, 0]]])
print(out)
