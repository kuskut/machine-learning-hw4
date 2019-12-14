from os import path

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ALPHA = 0.000001
THRESHOLD = 0.1
EPSILON = 0.000001


def load_dataset(file_name):
    """
    loads dataset and returns it
    :param file_name:
    :return:
    """
    file_location = path.join(path.abspath(path.dirname(__file__)), '../data', file_name)
    return np.loadtxt(file_location, dtype=np.float128)


def standardize_dataset(dataset):
    """
    standardizes a dataset
    :param x:
    :return:
    """
    x = dataset[:, :-1]
    dataset[:, :-1] = (x - x.mean(axis=0)) / np.std(x, axis=0)
    return dataset


def logplus_dataset(dataset):
    """
    computes log(a + 0.1) on all elements of input
    :param x:
    :return:
    """
    x = dataset[:, :-1]
    dataset[:, :-1] = np.log(x + 0.1)
    return dataset


def binarize_dataset(x):
    """
    if a > 0 => that element will become 1 else => 0
    :param x:
    :return:
    """
    x = dataset[:, :-1]
    binary_x = x.copy()
    binary_x[binary_x > 0] = 1
    binary_x[binary_x <= 0] = 0
    dataset[:, :-1] = binary_x
    return dataset


def h(x, beta):
    """
    hypothesis function (sigmoid of input x matrix with beta parameters)
    :param x: i*j matrix
    :param beta: j*1 vector
    :return: i*1 vector
    """
    return 1 / (1 + np.exp(-x.dot(beta)))


def compute_descent_size(x, y, beta):
    """
    computes the amount of descent for each parameter
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: j*1 vector
    """
    return ((h(x, beta) - y).T.dot(x)).T


def gradient_descent_step(x, y, beta):
    """
    discends beta parameters for a single step and returns the result
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: j*1 vector (new beta values)
    """

    return beta - (ALPHA * compute_descent_size(x, y, beta))


def cost(x, y, beta):
    """
    computes cost of logistic model based on current beta values
    :param x: i*j matrix
    :param y: i*1 vector
    :param beta: j*1 vector
    :return: number
    """

    h_value = h(x, beta)

    s1 = - (y.T.dot(np.log(h_value + EPSILON)))
    s2 = - ((1 - y).T.dot(np.log(1 - h_value + EPSILON)))

    return (s1 + s2)[0, 0]


def gradient_descent(x, y):
    """
    computes beta using gradient descent algorithm and computes list of costs
    :param x: i*j
    :param y: i*1
    :return: tuple containing beta (j*1 vector), costs list (list of costs in each iteration)
    """
    beta = np.zeros((x.shape[1], 1))

    costs_list = []
    last_cost = math.inf
    current_cost = 0

    while abs(last_cost - current_cost) > THRESHOLD:
        beta = gradient_descent_step(x, y, beta)

        last_cost = current_cost
        current_cost = cost(x, y, beta)

        costs_list.append(current_cost)

    return beta, costs_list


def draw_costs_plot(costs_list):
    sns.lineplot(range(0, len(costs_list)), costs_list)
    plt.show()


dataset = load_dataset('spam.data')
split_boundary = math.floor(80 * dataset.shape[0] / 100)

# apply a function on dataset feature elements (uncomment any one you want)
dataset = standardize_dataset(dataset)
# dataset = logplus_dataset(dataset)
# dataset = binarize_dataset(dataset)

# np.random.shuffle(dataset)
training_dataset, test_dataset = dataset[:split_boundary], dataset[split_boundary:]

# attach a column of 1s to the beginning of x
x = dataset[:, :-1]
x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x))

# save targets in separate variables
y = dataset[:, -1].reshape((x.shape[0], 1))

# get model information
beta, costs_list = gradient_descent(x, y)

# draw cost plot
draw_costs_plot(costs_list)
