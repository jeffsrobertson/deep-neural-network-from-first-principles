import numpy as np


class NeuralNetwork():

    def __init__(self, loss='crossentropy', classifier='categorical', metric='accuracy', learning_rate=.04):

        ALLOWED_LOSSES = ['crossentropy']
        ALLOWED_CLASSIFIERS = ['binary', 'categorical']
        ALLOWED_METRICS = ['accuracy']
        assert loss in ALLOWED_LOSSES, "Invalid loss function {}. " \
                                       "Allowed loss functions: ".format(loss)+", ".join(ALLOWED_LOSSES)
        assert classifier in ALLOWED_CLASSIFIERS, "Invalid classifier {}. " \
                                                  "Allowed classifiers: ".format(classifier)+", ".join(ALLOWED_CLASSIFIERS)
        assert metric in ALLOWED_METRICS, "Invalid metric {}. " \
                                       "Allowed metrics: ".format(metric)+", ".join(ALLOWED_METRICS)

        self.W = [np.NaN]
        self.b = [np.NaN]
        self.Z = [np.NaN]
        self.A = [np.NaN]

        self.layer_activations = [None]
        self.layer_units = [0]
        self.layer_initializers = [None]

        self.loss_function = loss
        self.classifier = classifier
        self.metric = metric
        self.loss_history = [np.NaN]
        self.metric_history = [np.NaN]
        self.loss = 0.
        self.learning_rate = learning_rate

    def activation_func(self, z, activation=None, return_derivative=False):
        '''
        Applies the activation function of a given layer to its weighted sums z

        :param z: Weighted sums of neurons in this layer, of size (n,m)
        :param activation: Designated activation function for this layer. Options are:
            'sigmoid'
            'tanh'
            'linear'
            'reLU'
            'softmax' : This is usually the last layer's activation in categorical classification NNs
        :param return_derivative: Set to True to return the activation function's derivative instead (used in
            calculating backpropagation)
        :return: Array of size (n, m)
        '''
        if activation == 'sigmoid':
            s = 1/(1 + np.exp(-z))
            if return_derivative:
                return s*(1 - s)
        elif activation == 'tanh':
            s = np.tanh(z)
            if return_derivative:
                return 1 - s**2
        elif activation == 'linear':
            s = z
            if return_derivative:
                return np.ones_like(z, dtype=np.float32)
        elif activation == 'reLU':
            s = np.maximum(z, 0)
            if return_derivative:
                return (s > 0)*s
        elif activation == 'softmax':
            max_z = np.max(z, axis=0, keepdims=True)
            s = np.exp(z - max_z)
            norm = np.sum(s, axis=0, keepdims=True)
            s = s/norm
            if return_derivative:
                return s*(1-s)
        else:
            raise ValueError('Invalid activation function: {}'.format(activation))
            return False
        return s

    def forward_propagate(self):
        num_layers = len(self.layer_units) - 1
        for l in range(1, num_layers+1):
            self.Z[l] = np.dot(self.W[l], self.A[l-1]) + self.b[l]
            self.A[l] = self.activation_func(self.Z[l], activation=self.layer_activations[l])

    def calculate_output_dZ(self, actual):
        '''
        Calculates the gradient of the loss function with respect to Z (i.e. dL/dz) in the last layer of the NN.
        This is the first step required in beginning back propagation.

        This is favorable to calculating dA first and then dZ, because it avoids potential divide by zero errors.

        :param actual: ground truth array of training set.
                For categorical classifier, this array must be one-hot coded, and of size (N, m)
                For binary classification, this array should be size (1, m)
        :return dZ: dZ array of last layer. Size is same as input array.
        '''
        if self.classifier=='categorical':
            assert self.layer_activations[-1] == 'softmax', "For categorical classification, " \
                                                            "last layer's activation MUST be 'softmax'"
            if self.loss_function=='crossentropy':
                return -actual*(1 - self.A[-1])
        elif self.classifier=='binary':
            assert self.layer_activations[-1] == 'sigmoid', "For binary classification, " \
                                                            "last layer's activation MUST be 'sigmoid'"
            assert self.layer_units[-1] == 1, "For binary classification, last layer must only have 1 unit"
            if self.loss_function=='crossentropy':
                return -actual + self.A[-1]

    def backward_propagate(self, actual):

        m = actual.shape[-1]
        num_layers = len(self.layer_units) - 1

        # Iterate from layer L-1 to layer 1
        for l in range(num_layers, 0, -1):
            if l == num_layers:
                dZ = self.calculate_output_dZ(actual)
            else:
                gprime = self.activation_func(self.Z[l], activation=self.layer_activations[l], return_derivative=True)
                dZ = gprime*np.dot(np.transpose(self.W[l+1]), dZ) # Calculates dZ from previous layer's W and dZ

            dW = (1/m)*np.dot(dZ, np.transpose(self.A[l-1]))
            dB = (1/m)*np.sum(dZ, axis=1, keepdims=True)

            # Update parameters
            self.W[l] = self.W[l] - self.learning_rate*dW
            self.b[l] = self.b[l] - self.learning_rate*dB

    def backward_propagate_old(self, actual):

        m = actual.shape[-1]
        num_layers = len(self.layer_units) - 1

        # Iterate through layers, starting at layer L and ending at layer 1
        dA_prev = self.calculate_loss(actual, return_derivative=True)
        for l in range(num_layers, 0, -1):

            # Calculate dZ of this layer
            gprime = self.activation_func(self.Z[l], activation=self.layer_activations[l], return_derivative=True)
            dZ = gprime*dA_prev

            # Calculate backprop derivative terms
            dW = (1/m)*np.dot(dZ, np.transpose(self.A[l-1]))
            dB = np.sum(dZ, axis=1, keepdims=True)

            # Update weights/biases
            self.W[l] = self.W[l] - self.learning_rate*dW
            self.b[l] = self.b[l] - self.learning_rate*dB

            # Calculate dA for the next layer
            dA_prev = np.dot(np.transpose(self.W[l]), dZ)

    def calculate_loss(self, actual):
        '''
        Calculates the desired loss function with respect to the NN's outputted prediction in the final layer.

        Current options for loss are:
            'crossentropy'

        :param actual: ground truth array.

                For categorical classification, this is of size (N, m), where N = # categories
                and m = # training samples. Note that this array MUST be one-hot coded!!

                For binary classification, this is of size (1, m). Does not need to be one-hot coded

        :return: Scalar value representing the total (summed) loss.
                 If return_derivative=True, returns an array of size (1, m) representing the derivative of the loss of
                 each sample
        '''

        m = actual.shape[-1]
        predicted = self.A[-1]
        assert actual.shape == predicted.shape, "Ground truth array {} must be same size as output " \
                                                "of final layer {}".format(actual.shape, predicted.shape)

        if self.loss_function == 'crossentropy' and self.classifier == 'binary':
            assert self.layer_activations[-1] == 'sigmoid' and predicted.shape[0] == 1, \
                'For binary crossentropy, final layer must have a single neuron with sigmoid activation.'
            loss_vector = -actual*np.log(predicted) - (1 - actual)*np.log(1-predicted)
        elif self.loss_function == 'crossentropy' and self.classifier == 'categorical':
            assert self.layer_activations[-1] == 'softmax', "For categorical crossentropy, final layer " \
                                                            "must have a softmax activation"
            loss_vector = np.sum(-actual*np.log(predicted), axis=0, keepdims=True)
        else:
            raise ValueError('Invalid loss function: {}'.format(self.loss_function))

        self.loss = (1/m)*np.sum(loss_vector)
        self.loss_history.append(self.loss)

    def initialize_params(self, X_input):
        '''
        Initializes weight and bias matrices for training, using the desired initialization algorithm(s).

        :param X_input: Training set, of size (# params, # samples)
        '''

        self.layer_units[0] = X_input.shape[0]
        m = X_input.shape[-1]

        num_layers = len(self.layer_units) - 1
        for l in range(1, num_layers+1):
            n_l = self.layer_units[l]
            n_prev = self.layer_units[l-1]

            if self.layer_initializers[l] == 'zeros':
                self.W.append(np.zeros(shape=(n_l, n_prev), dtype=np.float32))
                self.b.append(np.zeros(shape=(n_l, 1), dtype=np.float32))
            elif self.layer_initializers[l] == 'random':
                self.W.append(np.random.rand(n_l, n_prev).astype(np.float32))
                self.b.append(np.random.rand(n_l, 1).astype(np.float32))
            elif self.layer_initializers[l] == 'xavier':
                stddev = np.sqrt(1/n_prev)
                self.W.append(stddev*np.random.randn(n_l, n_prev).astype(np.float32))
                self.b.append(stddev*np.random.randn(n_l, 1).astype(np.float32))
            else:
                print('>> Error: Invalid activation in layer {}'.format(l))
                return False
            self.Z.append(np.zeros(shape=(n_l, m), dtype=np.float32))
            self.A.append(np.zeros(shape=(n_l, m), dtype=np.float32))

    def add_layer(self, units, activation='tanh', initializer='xavier'):
        '''
        Add a fully connected layer to the NN

        :param units: number of neurons in this layer
        :param activation: Activation function for the neurons in this layer. Current options are:
            'tanh'
            'sigmoid'
            'linear'
            'reLU'
            'sigmoid'
        :param initializer: Algorithm to initialize weights/biases of the NN. Current options are:
            'zeros' : Sets all weights/biases to zero
            'ones' : Sets all weights/biases to one
            'random' : Sets weights/biases to random numbers between 0 and 1, from a uniform distribution
            'xavier' : Sets weights/biases to numbers sampled from a normal distribution, with variance = 1/n_{l-1}
        '''

        self.layer_units.append(units)
        self.layer_activations.append(activation)
        self.layer_initializers.append(initializer)

    def train(self, x_train, y_train, iterations=1):
        '''
        Train the weights/biases of the neural network over desired number of iterations

        :param x_train: training set array, of size (n, m), where n is number of features and m is number of training samples
        :param y_train: ground truth array of training set, of size (m), or (n, m) (if one-hot encoded)
        :param iterations: number of times that the NN performs forward/backward prop for training
        '''

        # Make sure ground truth arrays are correct size
        # Binary classification should be size (1, m)
        # Categorical classification should be one-hot encoded and size (N, m)
        if self.classifier == 'categorical':
            if y_train.ndim == 1:
                one_hot_array = np.zeros(shape=(y_train.max()+1, y_train.size))
                one_hot_array[y_train, np.arange(y_train.size)] = 1
                y_train = one_hot_array
        if self.classifier == 'binary':
            if y_train.ndim == 1:
                y_train = np.expand_dims(y_train, axis=0)

        assert x_train.shape[-1] == y_train.shape[-1], 'x_train {} and y_train {} have different sized ' \
                                                       'training samples'.format(x_train.shape, y_train.shape)

        self.A[0] = x_train
        self.initialize_params(x_train)

        print('Training neural network.')
        for i in range(iterations):

            self.forward_propagate()
            self.calculate_loss(y_train)
            self.backward_propagate(y_train)

            print_step_size = int(iterations/100) if iterations > 100 else 1
            if i % print_step_size == 0:
                metric = self.calculate_metric(self.A[-1], y_train)
                self.metric_history.append(metric)
                print('At iteration {}, J = {}. {} = {:.2f}%'.format(i, self.loss, self.metric, 100*metric))

    def test(self, x_test, y_test):
        '''
        Runs the NN on x_test and compares the predicted result to the given ground truth y_test

        :param x_test: test set, of size (n, m)
        :param y_test: ground truth of test set, of size (1, m)
        :return: The NN's prediction of x_test. Size is (L, m), where L is number of neurons in final layer
        '''

        if y_test.ndim == 1:
            y_test = np.expand_dims(y_test, axis=0)
        assert x_test.shape[-1] == y_test.shape[-1], 'x_test and y_test have different number of features'
        assert x_test.shape[0] == self.A[0].shape[0], 'x_test has different number of features than what the NN was trained on'

        # Make sure ground truth arrays are correct size
        # Binary classification should be size (1, m)
        # Categorical classification should be one-hot encoded and size (N, m)
        if self.classifier == 'categorical':
            if y_test.ndim == 1:
                one_hot_array = np.zeros(shape=(y_test.max()+1, y_test.size))
                one_hot_array[y_test, np.arange(y_test.size)] = 1
                y_test = one_hot_array
        if self.classifier == 'binary':
            if y_test.ndim == 1:
                y_test = np.expand_dims(y_test, axis=0)

        assert x_test.shape[-1] == y_test.shape[-1], 'x_train {} and y_train {} have different sized ' \
                                                       'training samples'.format(x_train.shape, y_train.shape)

        # Set test set to the 0th layer
        self.A[0] = x_test

        self.forward_propagate()

        return self.A[-1]

    def calculate_metric(self, prediction, actual):
        '''
        Calculates the desired metric parameter

        :param prediction: prediction array of size (n, m) for categorical or (1, m) for binary
        :param metric: Can be one of the following:
            'accuracy' : (correct predictions)/(total predictions)
            (todo: add precision, recall, F1 score)

        :return: Scalar value of metric
        '''

        prediction_onehot = np.zeros_like(prediction)
        prediction_onehot[prediction.argmax(0), np.arange(prediction.shape[1])] = 1

        if self.metric == 'accuracy':
            correct_predictions = np.sum(np.logical_and(prediction_onehot, actual))
            total_predictions = prediction_onehot.shape[-1]
            return correct_predictions/total_predictions
        else:
            raise ValueError('Invalid metric: {}'.format(self.metric))


if __name__ == '__main__':

    # Initialize the NN and build the layers
    nn = NeuralNetwork(loss='crossentropy', classifier='categorical', learning_rate=.02, metric='accuracy')
    nn.add_layer(units=20, activation='tanh', initializer='xavier')
    nn.add_layer(units=40, activation='tanh', initializer='xavier')
    nn.add_layer(units=10, activation='softmax', initializer='xavier')

    # Load in training/test data, reshape it to be compatible with neural network
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape training/test data so that they're the shape that the NN expects
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T

    # Train the network
    nn.train(x_train=x_train, y_train=y_train, iterations=150)

    # Test NN on our test set
    prediction = nn.test(x_test=x_test, y_test=y_test)

    import matplotlib.pyplot as plt
    plt.plot(nn.loss_history)
    plt.title('Cost function vs # iterations')
    plt.xlabel('# iterations')
    plt.ylabel('J')

    plt.figure(2)
    plt.plot(100*np.arange(len(nn.metric_history)), nn.metric_history)
    plt.title('Accuracy vs # iterations')
    plt.ylabel('%')
    plt.xlabel('# iterations')
    plt.show()

