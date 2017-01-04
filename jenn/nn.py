import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from PIL import Image
import multiprocessing
import ctypes
import random

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def build_input_data():
    X = []
    y = []
    unicode_A = ord('A')
    counter = 0
    paths = os.listdir('renders')
    random.shuffle(paths)
    for f in paths:
        if os.path.isfile(os.path.join('renders', f)):
            if f[-6:] == '_0.png' and counter % 10 == 0: # counter % 50 == 0:
                X.append(np.array(Image.open('renders/' + f), dtype=float).flatten())
                new_y_row = np.zeros(26, dtype=float)
                new_y_row[ord(f[:1]) - unicode_A] = 1.0
                y.append(new_y_row)
            counter += 1
    # Normalize -- do I need to normalize y?
    denominator = np.amax(X, axis=0);
    X = X/denominator


    # split the datasets in half for training and evaluation
    midway = len(X)/2
    training_X    = X[:midway]
    evaluation_X  = X[midway + 1:]
    training_y    = y[:midway]
    evaluation_y  = y[midway + 1:]

    return (training_X, training_y), (evaluation_X, evaluation_y)

# global data
CONCURRENCY = 8
TRAIN_DATA, EVAL_DATA = build_input_data()

# looks like
# [
#     [xgroup1, ygroup1],
#     [xgroup2, ygroup2],
#     [xgroupN, ygroupN] // where N is concurrency
# ]
# the point of splitting  groups is that
# each group can be processed concurrently
XY_TRAINING_GROUPS = zip(
    list(chunks(TRAIN_DATA[0], len(TRAIN_DATA[0])/CONCURRENCY)),
    list(chunks(TRAIN_DATA[1], len(TRAIN_DATA[0])/CONCURRENCY)))

# object that shows the internal state and output of a single nn evaluation
# nn.forward(X) returns this
NeuralNetRun = namedtuple(
    'NeuralNetRun', ['X', 'z2', 'a2', 'z3', 'yHat', 'W1', 'W2', 'b1', 'b2'])

NeuralNet = namedtuple('NeuralNet', ['W1', 'W2', 'b1', 'b2'])

def costFunction(y, yHat):
    if yHat.any() < 0 or yHat.any() > 1:
        print "Bad yHat detected."
    sum = 0
    for m in range(len(yHat)):
        for n in range(26):
            sum += (y[m][n])*np.log(yHat[m][n]) + (1-y[m][n])*np.log(1-yHat[m][n])
    cost = sum * -1 / len(yHat)
    return cost

def costFunctionPrime(netRun, y):
    def sigmoidPrime(z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    delta3 = netRun.yHat - y
    delta2 = np.dot(delta3, netRun.W2.T)*sigmoidPrime(netRun.z2)
    dJdW2 = np.dot(netRun.a2.T, delta3)
    dJdW1 = np.dot(netRun.X.T, delta2)
    dJdb2 = np.sum(delta3, axis=0, keepdims=True)
    dJdb1 = np.sum(delta2, axis=0, keepdims=True)

    # OG Andrew's version.
    # delta3 = np.multiply(-(y-netRun.yHat), sigmoidPrime(netRun.z3))
    # dJdW2 = np.dot(netRun.a2.T, delta3)
    # delta2 = np.dot(delta3, netRun.W2.T)*sigmoidPrime(netRun.z2)
    # dJdW1 = np.dot(netRun.X.T, delta2)
    # dJdb2 = np.sum(delta3, axis=0, keepdims=True)
    # dJdb1 = np.sum(delta2, axis=0, keepdims=True)

    return dJdW1, dJdW2, dJdb1, dJdb2


def forward(NN, X):
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    z2 = np.dot(X, NN.W1) + NN.b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, NN.W2) + NN.b2
    # Softmax. This makes cost function graph go up + to the right.
    # z3_exp = np.exp(z3)
    # yHat = z3_exp / np.sum(z3_exp)
    yHat = sigmoid(z3)
    yHat = yHat / np.sum(yHat)
    return NeuralNetRun(X, z2, a2, z3, yHat, NN.W1, NN.W2, NN.b1, NN.b2)

def build_neural_net(inputLayerSize=100,
        outputLayerSize=26, hiddenLayerSize=20):
    np.random.seed(0)
    W1 = np.random.randn(inputLayerSize, hiddenLayerSize)
    W2 = np.random.randn(hiddenLayerSize, outputLayerSize)
    b1 = np.zeros((1, hiddenLayerSize))
    b2 = np.zeros((1, outputLayerSize))
    return NeuralNet(W1, W2, b1, b2)

# this is the chunk of code that should be serialized and
# sent to each process, can only take a single argument
def concurrent(args):
    NN, i = args

    X,y = XY_TRAINING_GROUPS[i]

    # run input through the nn
    netRun = forward(NN, X)

    # calculate gradient
    dJdW1, dJdW2, dJdb1, dJdb2 = costFunctionPrime(netRun, y)

    return dJdW1, dJdW2, dJdb1, dJdb2, netRun.yHat

def single_training_iteration(NN, scalar, pool):

    if CONCURRENCY == 1:
        # easier to debug
        results = []
        for i in range(len(XY_TRAINING_GROUPS)):
            results.append(concurrent((NN, i)))
    else:
        results = pool.map(concurrent, [(NN, i) for i in range(len(XY_TRAINING_GROUPS))])

    # add all the results together for each group as if we had
    # run them all together
    dJdW1, dJdW2, dJdb1, dJdb2, first_yHat = results[0]
    yHat = first_yHat
    for dJdW1_current, dJdW2_current, dJdb1_current, dJdb2_current, yHat_current in results[1:]:
        dJdW1=np.add(dJdW1, dJdW1_current)
        dJdW2=np.add(dJdW2, dJdW2_current)
        dJdb1=np.add(dJdb1, dJdb1_current)
        dJdb2=np.add(dJdb2, dJdb2_current)
        yHat=np.concatenate([yHat, yHat_current])

    new_W1 = NN.W1 - scalar * dJdW1
    new_W2 = NN.W2 - scalar * dJdW2
    new_b1 = NN.b1 - scalar * dJdb1
    new_b2 = NN.b2 - scalar * dJdb2

    return NeuralNet(new_W1, new_W2, new_b1, new_b2), yHat

# run all the iterations
def train(initialNeuralNet, scalar=0.001, num_iterations=700):

    yHat_history = []

    pool = multiprocessing.Pool(CONCURRENCY)

    current_NN = initialNeuralNet
    for i in range(0, num_iterations):
        result_NN, yHat = single_training_iteration(current_NN, scalar, pool)
        yHat_history.append(yHat)
        current_NN = result_NN

    cost_history = []
    for yHat in yHat_history:
        cost_history.append(costFunction(TRAIN_DATA[1], yHat))

    return current_NN, cost_history

def eval(name, NN, X, y):
    run = forward(NN, X)
    yHatIndices = np.argmax(run.yHat, axis=1)
    yIndicies = np.argmax(y, axis=1)

    differences = []
    for a, b in zip(yHatIndices, yIndicies):
        differences.append(a-b)

    results = []
    for value in differences:
        if value != 0:
            results.append(1)
        else:
            results.append(0)
    avg = float(sum(results))/float(len(results))
    correct = [r for r in results if r == 0]
    incorrect = [r for r in results if r == 1]
    print "{}: correct: {}, incorrect: {}, avg: {}".format(
        name, len(correct), len(incorrect), avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a NN')
    parser.add_argument('--outputPath', type=str)
    args = parser.parse_args()

    trainedNeuralNet, cost_history = train(initialNeuralNet=build_neural_net())

    eval("training data", trainedNeuralNet, TRAIN_DATA[0], TRAIN_DATA[1])
    eval("evaluation data", trainedNeuralNet, EVAL_DATA[0], EVAL_DATA[1])

    y = cost_history
    x = range(0, len(cost_history))
    plt.scatter(x, y)
    plt.show()
