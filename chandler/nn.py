from scipy import optimize
import numpy as np
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


def build_neural_net(inputLayerSize=2,
        outputLayerSize=1, hiddenLayerSize=3):
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
    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)
    # Normalize
    X = X/np.amax(X, axis=0)
    y = y/100
    return X, y

X, y = build_test_input()

initial_net = build_neural_net() # random weights

print forward(initial_net, [3,5]).yHat

trained_net, iteration_cost_history = train(initial_net, X, y)

print forward(trained_net, [3,5]).yHat

plt.plot(iteration_cost_history)
plt.grid(1)
plt.show()


