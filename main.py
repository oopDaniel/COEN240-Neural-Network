import numpy as np
from tensorflow.keras.datasets import mnist
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')

def ReLU(A):
    return A * (A > 0)

def ReLU_derivative(A):
    return 1. * (A > 0)

def softmax(A):
    # Use Min-max scaling to normalize data to prevent overflow encountered in exp
    mms = MinMaxScaler((-200, 200))
    mms.fit(A)
    A = mms.transform(A)

    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def softmax_error_cross_entropy_derivative(prediction, label):
    return prediction - label


class NeuralNetwork:
    """
    Predefind 2 layers neural network
    Hidden layer: ReLU activation function
    Output layer: softmax activation function
    """
    def __init__(
        self,
        layers,
        learning_rate=10e-4,
        epochs=100,
        error_function='cross_entropy'
    ):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.error_function = error_function

    def fit(self, data, label):
        attributes = data.shape[1]
        self._init_params(attributes)

        # mi, ma = data.min(), data.max()
        # data = (data - mi) / (ma - mi)

        label = self._get_label_matrix(label)
        print(label)
        self._train(data, label)

    """
    Initialize random weight and bias according to the structure of network
    """
    def _init_params(self, attributes):
        layer_nodes = list(map(lambda l : l[0], self.layers))
        prev = attributes
        weights = []
        for nodes in layer_nodes:
            weights.append({
                'weight': np.random.rand(prev, nodes),
                'bias': np.random.randn(nodes)
            })
            prev = nodes

        self.weights = weights

    """
    Convert label(Dx1) into label matrix(DxM) where D = data and M =
    number of classes to classify.
    """
    def _get_label_matrix(self, label):
        output_nodes_count = self.layers[-1][0]
        label_matrix = np.zeros((label.size, output_nodes_count))
        for index, value in enumerate(label):
            label_matrix[index][value] = 1
        return label_matrix

    """
    Train the neural network using the given definition and epochs
    """
    def _train(self, data, label):
        total_layers = len(self.layers)
        output_layer = total_layers - 1
        learning_rate = self.learning_rate
        error_function = self.error_function

        for epoch in range(self.epochs):
        # for epoch in range(1):
            #### Feed Forward
            output = [0] * total_layers
            costs = [0] * total_layers
            prev = data

            for i in range(total_layers):
                weights = self.weights[i]
                z = np.dot(prev, weights['weight']) + weights['bias']

                activation_function = self.layers[i][1]
                print('- in layer', i, 'activate with ', activation_function)

                if activation_function == 'ReLU':
                    a = ReLU(z)
                elif activation_function == 'Softmax':
                    print('softmax')
                    a = softmax(z)
                # NOTE: Define other activation function if needed

                output[i] = { 'z': z, 'a': a }
                prev = a

            #### Back Propagation
            for i in range(total_layers):
                i = total_layers - i - 1
                activation_function = self.layers[i][1]

                # Output layer doesn't have an error term 𝛿 from next layer
                if i == output_layer:
                    dcost_dah = 1
                else:
                    dcost_dah = np.dot(costs[i + 1]['error'] , self.weights[i + 1]['weight'].T)

                # Derivative of cost function
                if i == output_layer:
                    # Define others if needed
                    if activation_function == 'Softmax' and error_function == 'cross_entropy':
                        dah_dzh = softmax_error_cross_entropy_derivative(output[i]['a'], label)
                else:
                    if activation_function == 'ReLU':
                        dah_dzh = ReLU_derivative(output[i]['z'])

                # Activated output from prev layer
                # Use raw data in 1st hidden layer
                dzh_dwh = output[i - 1]['a'] if i > 0 else data

                dcost_weight = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
                dcost_bias = dcost_weight if i == output_layer else dcost_dah * dah_dzh

                costs[i] = { 'weight': dcost_weight, 'bias': dcost_bias, 'error': dah_dzh }

            #### Update Weight
            for i in range(total_layers):
                self.weights[i]['weight'] -= learning_rate * costs[i]['weight']
                self.weights[i]['bias'] -= learning_rate * costs[i]['bias'].sum(axis=0)

            print(output[-1]['a'].shape)
            print(label.shape)
            print('Loss function value: ', np.sum(-label * np.log(output[-1]['a'])))

if __name__ == '__main__':
    # Load data
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    # Reshape data
    train_data = np.reshape(train_data, (60000, 28 * 28)).astype(np.float64)
    test_data = np.reshape(test_data, (10000, 28 * 28)).astype(np.float64)
    train_data, test_data = train_data / 255.0, test_data / 255.0

    # print(train_data.shape)
    # Define network structure
    layers = [(512, 'ReLU'), (10, 'Softmax')]

    # Start training
    net = NeuralNetwork(layers, epochs=5, error_function='cross_entropy')
    net.fit(train_data, train_label)

#   # Build logistic regression model
#   log_reg = LogisticRegression(
#     solver='saga',
#     multi_class='multinomial',
#     max_iter=100,
#     verbose=2
#   )

#   # Train the model
#   log_reg.fit(train_data, train_label)

#   # Predict the result of test data
#   predition = log_reg.predict(test_data)

#   # Show the accuracy and confusion matrix
#   print(metrics.accuracy_score(test_label, predition))
#   print(metrics.confusion_matrix(test_label, predition))