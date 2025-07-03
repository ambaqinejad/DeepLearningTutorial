import random

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], 'w1 =', '%5.2f' % w[1], 'w2 =', '%5.2f' % w[2])

random.seed(7)
LEARNING_RATE = .1
index_list = [0, 1, 2, 3]  # used to randomize order

# [bias, x1, x2]
x_train = [(1.0, -1.0, -1.0),
           (1.0, -1.0, 1.0),
           (1.0, 1.0, -1.0),
           (1.0, 1.0, 1.0)]

y_train = [1.0, 1.0, 1.0, -1.0]

# random number
w = [.2, -.6, .25]

show_learning(w)


def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z = z + w[i] * x[i]

    # sign function
    if z < 0:
        return -1
    else:
        return 1