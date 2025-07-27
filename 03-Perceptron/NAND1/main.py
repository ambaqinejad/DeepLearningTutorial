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

all_correct = False

while not all_correct:
    random.shuffle(index_list)
    all_correct = True
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)
        if p_out != y:
            all_correct = False
            for j in range(len(w)):
                w[j] += y * LEARNING_RATE * x[j]
            show_learning(w)

print('y0_turth =', '%5.2f' % y_train[0], ', y0 =', '%5.2f' % compute_output(w, x_train[0]))
print('y1_turth =', '%5.2f' % y_train[1], ', y0 =', '%5.2f' % compute_output(w, x_train[1]))
print('y2_turth =', '%5.2f' % y_train[2], ', y0 =', '%5.2f' % compute_output(w, x_train[2]))
print('y3_turth =', '%5.2f' % y_train[3], ', y0 =', '%5.2f' % compute_output(w, x_train[3]))