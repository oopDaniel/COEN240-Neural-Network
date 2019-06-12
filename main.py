from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
from keras.datasets import mnist
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

# Reshape label matrices
labels, test_labels = to_categorical(train_label), to_categorical(test_label)

# Create model
model = Sequential()
model.add(Dense(train_data.shape[0], input_dim=train_data.shape[1], use_bias=False))
model.add(Dense(hidden_layer_nodes_count, activation='relu'))
model.add(Dense(output_nodes_count, activation='softmax'))

# Compile model
adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# Fit the model
model.fit(train_data, labels, epochs=5)

# Evaluate the model and show the accuracy and confusion matrix
prediction = model.predict_classes(test_data)
print(metrics.accuracy_score(test_label, prediction))
print(metrics.confusion_matrix(test_label, prediction))