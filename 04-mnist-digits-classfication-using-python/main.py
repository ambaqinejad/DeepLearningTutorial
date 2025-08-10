import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMAGE_FILENAME = "./dataset/train-images-idx3-ubyte/train-images-idx3-ubyte"
TRAIN_LABEL_FILENAME = "./dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
TEST_IMAGE_FILENAME = "./dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
TEST_LABEL_FILENAME = "./dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 20


def read_mnist():
    # Read files
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    # Reformat and Standardize
    X_train = train_images.reshape(60000, 784)
    mean = np.mean(X_train)
    stddev = np.std(X_train)
    X_train = (X_train - mean) / stddev
    X_test = test_images.reshape(10000, 784)
    X_test = (X_test - mean) / stddev

    # One-hot Encoded Output
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))

    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1

    return X_train, y_train, X_test, y_test


def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for i in range(neuron_count):
        for j in range(1, input_count + 1):
            weights[i][j] = np.random.uniform(-.1, .1)
    return weights


hidden_layer_w = layer_w(25, 784)
hidden_layer_y = np.zeros(25)
hidden_layer_error = np.zeros(25)

output_layer_w = layer_w(10, 25)
output_layer_y = np.zeros(10)
output_layer_error = np.zeros(10)

chart_x = []
chart_y_train = []
chart_y_test = []


def show_learning(epoch_no, train_acc, test_acc):
    global chart_x
    global chart_y_train
    global chart_y_test
    print(f"epoch no: {epoch_no}, training accuracy: {train_acc}, test accuracy: {test_acc}")
    chart_x.append(epoch_no + 1)
    chart_y_train.append(1.0 - train_acc)
    chart_y_test.append(1.0 - test_acc)


def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='training error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('Training Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def forward_pass(X):
    global hidden_layer_y
    global output_layer_y
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, X)
        hidden_layer_y[i] = np.tanh(z)
    hidden_output_array = np.concatenate(
        (np.array([1.0]), hidden_layer_y)
    )
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))


def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error
    for i, y in enumerate(output_layer_y):
        error_prime = -(y_truth[i] - y)
        derivative = y * (1.0 - y)
        output_layer_error[i] = error_prime * derivative
    for i, y in enumerate(hidden_layer_y):
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i + 1])
            error_weight_array = np.array(error_weights)
        derivative = 1.0 - y ** 2
        weighted_error = np.dot(error_weight_array,
                                output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative


def adjust_weights(X):
    global output_layer_w
    global hidden_layer_w
    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= X * LEARNING_RATE * error
    hidden_output_array = np.concatenate(
        (np.array([1.0]), hidden_layer_y)
    )
    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= (hidden_output_array*LEARNING_RATE*error)




X_train, y_train, X_test, y_test = read_mnist()
# Used for random order
index_list = list(range(len(X_train)))

for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0
    for j in index_list:
        X = np.concatenate((np.array([1.0]), X_train[j]))
        forward_pass(X)
        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training_results += 1
        backward_pass(y_train[j])
        adjust_weights(X)
    correct_test_results = 0
    for j in range(len(X_test)):
        X = np.concatenate((np.array([1.0]), X_test[j]))
        forward_pass(X)
        if output_layer_y.argmax() == y_test[j].argmax():
            correct_test_results += 1
    show_learning(i, correct_training_results/len(X_train), correct_test_results/len(X_test))
plot_learning()