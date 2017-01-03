import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from PIL import Image
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

# object that shows the internal state and output of a single nn evaluation
# nn.forward(X) returns this
NeuralNetRun = namedtuple('NeuralNetRun', ['X', 'z2', 'a2', 'z3', 'yHat', 'W1', 'W2'])

NeuralNet = namedtuple('NeuralNet', ['W1', 'W2'])

def costFunction(y, yHat):
    yHat_array = np.array(yHat.reshape(len(yHat), 26))
    sum = 0
    for m in range(len(yHat)):
        for n in range(26):
            sum += (y[m][n])*np.log(yHat[m][n]) + (1-y[m][n])*np.log(1-yHat[m][n])
    cost = sum * -1 / len(yHat)
    return cost


def costFunctionPrime(NN, netRun, y):
    def sigmoidPrime(z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    delta3 = netRun.yHat - y
    #delta3 = np.multiply(-(y-netRun.yHat), sigmoidPrime(netRun.z3))
    delta2 = np.dot(delta3, NN.W2.T)*sigmoidPrime(netRun.z2)
    dJdW2 = np.dot(netRun.a2.T, delta3)
    dJdW1 = np.dot(netRun.X.T, delta2)

    return dJdW1, dJdW2

def forward(NN, X):
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    z2 = np.dot(X, NN.W1)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, NN.W2)
    z3_exp = np.exp(z3)
    yHat = z3_exp / np.sum(z3_exp)
#    yHat = np.argmax(yHat_probs, axis=1)
    return NeuralNetRun(X, z2, a2, z3, yHat, NN.W1, NN.W2)

def build_neural_net(inputLayerSize=100,
        outputLayerSize=26, hiddenLayerSize=20):
    W1 = np.random.randn(inputLayerSize, hiddenLayerSize)
    W2 = np.random.randn(hiddenLayerSize, outputLayerSize)
    return NeuralNet(W1, W2)

# this is the chunk of code that should be serialized and
# sent to each process, can only take a single argument
def concurrent(args):
    NN, X, y = args

    # run input through the nn
    netRun = forward(NN, X)

    # calculate gradient
    dJdW1, dJdW2 = costFunctionPrime(NN, netRun, y)

    return dJdW1, dJdW2, netRun.yHat

def single_training_iteration(NN, scalar, X, y, concurrency):

    # Xygroups ooks like
    # [
    #     [xgroup1, ygroup1],
    #     [xgroup2, ygroup2]
    # ]
    # the point of splitting  groups is that
    # each group can be processed concurrently
    Xy_groups = zip(
        list(chunks(X, concurrency)),
        list(chunks(y, concurrency)))

    results = []
    for Xgroup, ygroup in Xy_groups:
        results.append(concurrent((NN, Xgroup, ygroup)))

    # add all the results together for each group as if we had
    # run them all together
    dJdW1, dJdW2, first_yHat = results[0]
    yHat = first_yHat
    for dJdW1_current, dJdW2_current, yHat_current in results[1:]:
        dJdW1=np.add(dJdW1, dJdW1_current)
        dJdW2=np.add(dJdW2, dJdW2_current)
        yHat=np.concatenate([yHat, yHat_current])

    new_W1 = NN.W1 - scalar * dJdW1
    new_W2 = NN.W2 - scalar * dJdW2

    return NeuralNet(new_W1, new_W2), yHat

def build_input_data():
    X = []
    y = []
    unicode_A = ord('A')
    counter = 0
    for f in os.listdir('renders'):
        if os.path.isfile(os.path.join('renders', f)):
            if counter % 50 == 0:
                X.append(np.array(Image.open('renders/' + f)).flatten())
                new_y_row = np.zeros(26)
                new_y_row[ord(f[:1]) - unicode_A] = 1
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

# run all the iterations
def train(initialNeuralNet, training_data, scalar=0.001, num_iterations=1000, concurrency=4):

    yHat_history = []

    X, y = training_data()

    current_NN = initialNeuralNet
    for i in range(0, num_iterations):
        result_NN, yHat = single_training_iteration(current_NN, scalar, X, y, concurrency)
        yHat_history.append(yHat)
        current_NN = result_NN

    cost_history = []
    for yHat in yHat_history:
        cost_history.append(costFunction(y, yHat))

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

    training_data, evaluation_data = build_input_data()

    trainedNeuralNet, cost_history = train(
        initialNeuralNet=build_neural_net(),
        training_data=lambda: training_data)

    eval("training data", trainedNeuralNet, training_data[0], training_data[1])
    eval("evaluation data", trainedNeuralNet, evaluation_data[0], evaluation_data[1])

    y = [y for y in cost_history]
    x = range(0, len(cost_history))
    plt.scatter(x, y)
    plt.show()
