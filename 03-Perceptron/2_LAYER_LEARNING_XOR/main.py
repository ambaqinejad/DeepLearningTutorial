import numpy as np

np.random.seed(3)

LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

# (bias, x1, x2)
X_train = [
    np.array([1.0, -1.0, -1.0]),
    np.array([1.0, -1.0, 1.0]),
    np.array([1.0, 1.0, -1.0]),
    np.array([1.0, 1.0, 1.0])
]

y_train = [0.0, 1.0, 1.0, 0.0]


def wight_generator(input_count):
    weights = np.zeros(input_count + 1)  # bias + 2 inputs
    for i in range(1, (input_count + 1)):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights


n_w = [wight_generator(2), wight_generator(2), wight_generator(2)]
n_y = [0, 0, 0]  # output of neurons
n_error = [0, 0, 0]


def show_learning():
    print("Current Weights: ")
    for i, w in enumerate(n_w):
        print('neuron ', i, ': w0 = ', '%5.2f' % w[0], ': w1 = ', '%5.2f' % w[1], ': w2 = ', '%5.2f' % w[2])
    print("---------------")


def forward_pass(X):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], X))
    n_y[1] = np.tanh(np.dot(n_w[1], X))
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))


def backward_pass(y_truth):
    global n_error
    error_prime = -(y_truth - n_y[2])  # Derivative of loss-func
    derivative = n_y[2] * (1.0 - n_y[2])  # Logistic Derivative
    n_error[2] = error_prime * derivative
    derivative = 1.0 - n_y[0] ** 2  # tanh derivative
    n_error[0] = n_w[2][1] * n_error[2] * derivative
    derivative = 1.0 - n_y[1] ** 2  # tanh derivative
    n_error[0] = n_w[2][2] * n_error[2] * derivative


def adjust_weights(X):
    global n_w
    n_w[0] -= (X * LEARNING_RATE * n_error[0])
    n_w[1] -= (X * LEARNING_RATE * n_error[1])
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])


all_correct = False
while not all_correct:
    all_correct = True
    np.random.shuffle(index_list)
    for i in index_list:
        forward_pass(X_train[i])
        backward_pass(y_train[i])
        adjust_weights(X_train[i])
        show_learning()
    for i in range(len(X_train)):
        forward_pass(X_train[i])
        print(f"x1 = {X_train[i][1]}, x2 = {X_train[i][2]}, y = {n_y[2]}")
        if ((y_train[i] < .5) and (n_y[2] >= .5)) or ((y_train[i] >= .5) and (n_y[2] < .5)):
            all_correct = False

