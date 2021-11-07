import random
import numpy as np


def fourier(x):
    a = np.abs(np.fft.fft(x))
    a[0] = 0
    return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Adeline(object):

    def __init__(self, num_of_inputs, iterations=1000, learning_rate=0.001, bias=False, sigm=False):
        self.num_of_inputs = num_of_inputs
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = np.random.random(2 * self.num_of_inputs)
        self.w0 = np.random.random()
        self.bias = bias
        self.errors = []
        self.sigm = sigm

    def train(self, training_x, training_y):
        for _ in range(self.iterations):
            e = 0
            z = zip(training_x, training_y)
            for x, y in random.choices(list(z)):
                x = np.concatenate([x, fourier(x)])
                out = self.output(x)
                #self.learning_rate = (2 / (np.linalg.norm(x) ** 2)) / 2
                a = np.dot(self.weights, x)
                self.weights += self.learning_rate * (y - out) * x
                if self.bias:
                    self.w0 += self.learning_rate * (y - out)
                e += (y - out) ** 2
            self.errors.append(e)

    def _activation(self, x):
        if self.sigm:
            x = 0.8 * x + 0.1
            return sigmoid(x)
        else:
            return x

    def output(self, x):
        if self.bias:
            return self._activation(np.dot(self.weights, x) + self.w0)
        else:
            return self._activation(np.dot(self.weights, x))
