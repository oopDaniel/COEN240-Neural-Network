from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from tensorflow.keras.datasets import mnist
import sklearn.metrics as metrics

# fix random seed for reproducibility
np.random.seed(7)

# Load data
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Reshape data
train_data = np.reshape(train_data, (60000, 28 * 28))
test_data = np.reshape(test_data, (10000, 28 * 28))
train_data, test_data = train_data / 255.0, test_data / 255.0

# Defined some initial params
hidden_layer_nodes_count = 512
output_nodes_count = 10


labels, test_labels = np.zeros((train_label.size, output_nodes_count)), np.zeros((test_label.size, output_nodes_count))
for index, value in enumerate(train_label):
    labels[index][value] = 1
for index, value in enumerate(test_label):
    test_labels[index][value] = 1

# Create model
model = Sequential()
model.add(Dense(train_data.shape[0], input_dim=train_data.shape[1], use_bias=False))
model.add(Dense(hidden_layer_nodes_count, activation='relu'))
model.add(Dense(output_nodes_count, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_data, labels, epochs=5)

# Evaluate the model and show the accuracy and confusion matrix
prediction = model.predict_classes(test_data)
print(metrics.accuracy_score(test_label, prediction))
print(metrics.confusion_matrix(test_label, prediction))