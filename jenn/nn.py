# Neural Networks Demystified
# Part 6: Training
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch


## ----------------------- Part 1 ---------------------------- ##
import numpy as np
from PIL import Image
from scipy import optimize
from collections import namedtuple
import matplotlib.pyplot as plt


# object that shows the internal state and output of a single nn evalutation
# nn.forward(X) returns this
NeuralNetRun = namedtuple('NeuralNetRun', ['X', 'z2', 'a2', 'z3', 'yHat', 'W1', 'W2'])

NeuralNet = namedtuple('NeuralNet', ['W1', 'W2'])

def costFunction(y, yHat):
    return 0.5*sum((y-yHat)**2)

def costFunctionPrime(N, netRun, y):
    def sigmoidPrime(z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    delta3 = np.multiply(-(y-netRun.yHat), sigmoidPrime(netRun.z3))
    dJdW2 = np.dot(netRun.a2.T, delta3)

    delta2 = np.dot(delta3, N.W2.T)*sigmoidPrime(netRun.z2)
    dJdW1 = np.dot(netRun.X.T, delta2)

    return dJdW1, dJdW2

def forward(N, X):
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    z2 = np.dot(X, N.W1)
    a2 = sigmoid(z2)
    z3 = np.dot(a2, N.W2)
    yHat = sigmoid(z3)
    return NeuralNetRun(X, z2, a2, z3, yHat, N.W1, N.W2)


def build_neural_net(inputLayerSize=100,
        outputLayerSize=26, hiddenLayerSize=20):
    W1 = np.random.randn(inputLayerSize, hiddenLayerSize)
    W2 = np.random.randn(hiddenLayerSize, outputLayerSize)
    return NeuralNet(W1, W2)

def train(N, X, y, scalar=3, num_iterations=20):
    cost_history = []

    for i in range(0, num_iterations):
        # run input through the nn
        netRun = forward(N, X)

        # record the cost of this run
        cost_history.append(costFunction(y, netRun.yHat))

        # calculate gradient
        dJdW1, dJdW2 = costFunctionPrime(N, netRun, y)

        # update the nn params
        newW1 = N.W1 - scalar * dJdW1
        newW2 = N.W2 - scalar * dJdW2

        # overwrite N with new net
        N = NeuralNet(newW1, newW2)

    # N is now trained, use responsibly
    return N, cost_history

def build_test_input():
    X = [np.array(Image.open('letters/' + str(x) + '.tif')).flatten() for x in range(0, 26)]
    # Normalize -- do I need to normalize y?
    X = X/np.amax(X, axis=0)
    y = np.array([x for x in range(0, 26)], dtype=float)/25
    return X, y


def main():
    # NN = Neural_Network()
    # T = trainer(NN)
    # T.train(X, y)
    # plt.plot(T.J)
    # plt.grid(1)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.show()
    # NN.forward(X)
    # images = []
    # for x in range(0, 26):
    #     im = Image.open('letters/' + str(x) + '.tif')
    #     imarray = np.array(im)
    #     images.append(imarray)
    # print images
    X, y = build_test_input()
    print X[0]
    initial_net = build_neural_net() # random weights

    print forward(initial_net, X[0]).yHat

    trained_net, iteration_cost_history = train(initial_net, X, y)

    print forward(trained_net, X[0]).yHat

    plt.plot(iteration_cost_history)
    plt.grid(1)
    plt.show()


if __name__ == "__main__":
    main()
