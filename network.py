import numpy as np
from fc_layer import FCLayer


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.weights_loaded = False  # Flag to track if weights are loaded

    # add layers to network

    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def save_weights(self, filename):
        weights_to_save = []

        for layer in self.layers:
            layer_weights = {}
            if hasattr(layer, 'weights'):
                layer_weights['weights'] = layer.weights
                layer_weights['weights_shape'] = layer.weights.shape
            if hasattr(layer, 'bias'):
                layer_weights['bias'] = layer.bias
                layer_weights['bias_shape'] = layer.bias.shape

            if layer_weights:
                weights_to_save.append(layer_weights)

        if len(weights_to_save) > 0:
            try:
                np.save(filename, weights_to_save)
                print(f"Weights saved to {filename}")
            except Exception as e:
                print(f"Error saving weights to {filename}: {str(e)}")
        else:
            print("No weights to save in the network.")

    def load_weights(self, filename):
        try:
            loaded_weights = np.load(filename, allow_pickle=True)

            # Initialize an index to keep track of the layer being loaded
            layer_index = 0

            for layer_weights in loaded_weights:
                # Check if the layer at the current index is a fully connected layer
                if isinstance(self.layers[layer_index], FCLayer):
                    # Load weights and biases into the fully connected layer
                    self.layers[layer_index].weights = layer_weights.get(
                        'weights', self.layers[layer_index].weights)
                    self.layers[layer_index].bias = layer_weights.get(
                        'bias', self.layers[layer_index].bias)

                    # Ensure the loaded weights match the shapes of the layer
                    expected_weights_shape = self.layers[layer_index].weights.shape
                    expected_bias_shape = self.layers[layer_index].bias.shape

                    if (layer_weights.get('weights_shape') != expected_weights_shape or
                            layer_weights.get('bias_shape') != expected_bias_shape):
                        raise ValueError(
                            f"Loaded weights shape mismatch for layer {layer_index}.")

                    layer_index += 1

            self.weights_loaded = True  # Set the flag to indicate weights are loaded
            print(f"Weights loaded from {filename}")
        except Exception as e:
            print(f"Error loading weights from {filename}: {str(e)}")

    # predict output for given input

    def predict(self, input_data):
        # sample dimension first

        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # train the network not using batch gradient descent
    def fit(self, x_train, y_train, epochs, learning_rate, sample_size):
        # sample dimension first

        samples = len(x_train)
        decay = learning_rate/epochs

        indices = np.random.permutation(samples)[:sample_size]
        x_sample = x_train[indices]
        y_sample = y_train[indices]

        # training loop
        for i in range(epochs):
            err = 0
            correct_count = 0
            for j in range(sample_size):
                # forward propagation
                output = x_sample[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # backward propagation
                error = self.loss_prime(y_sample[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

                # compute loss for display
                err += self.loss(y_sample[j], output)
                # Assuming output is a vector of probabilities
                predicted_class = np.argmax(output)
                target_class = np.argmax(y_sample[j])
                # For display purposes only
                if predicted_class == target_class:
                    correct_count += 1

            # calculate average error on all samples
            err /= samples
            accuracy = correct_count / sample_size
            print('epoch %d/%d error=%f accuracy=%.4f' %
                  (i + 1, epochs, err, accuracy))
            # learning scheduler
            learning_rate *= 1/(1+decay*i)

    # train the network using batch gradient descent
    def batch_fit(self, x_train, y_train, epochs, learning_rate, batch_size):
        # sample dimension first
        # can set samples to amount if we don't want to train on the full dataset
        samples = len(x_train)
        decay = learning_rate / epochs

        # training loop
        for i in range(epochs):
            err = 0
            correct_count = 0

            # Shuffle the training data
            indices = np.random.permutation(samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            for j in range(0, samples, batch_size):
                x_batch = x_train_shuffled[j:j + batch_size]
                y_batch = y_train_shuffled[j:j + batch_size]
                batch_size_actual = len(x_batch)

                # Initialize gradients for this batch
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.weights_gradient = np.zeros_like(layer.weights)
                    if hasattr(layer, 'bias'):
                        layer.bias_gradient = np.zeros_like(layer.bias)

                for k in range(batch_size_actual):
                    # forward propagation
                    output = x_batch[k]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # backward propagation
                    error = self.loss_prime(y_batch[k], output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(
                            error, learning_rate)

                    # compute loss for display
                    err += self.loss(y_batch[k], output)

                    # Assuming output is a vector of probabilities
                    predicted_class = np.argmax(output)
                    target_class = np.argmax(y_batch[k])
                    # For display purposes only
                    if predicted_class == target_class:
                        correct_count += 1

                # Update weights and biases for this batch
                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer.weights -= (learning_rate /
                                          batch_size_actual) * layer.weights_gradient
                    if hasattr(layer, 'bias'):
                        layer.bias -= (learning_rate /
                                       batch_size_actual) * layer.bias_gradient

            # calculate average error on all samples
            err /= samples
            accuracy = correct_count / samples
            print('epoch %d/%d error=%f accuracy=%.4f' %
                  (i + 1, epochs, err, accuracy))
            # learning rate scheduler
            learning_rate *= 1 / (1 + decay * i)
