import numpy as np


def fourier(x):
    a = np.abs(np.fft.fft(x))
    a[0] = 0
    return a


class Adeline(object):

    def __init__(self, num_of_inputs, iterations=1000, learning_rate=0.5):
        self.num_of_inputs = num_of_inputs
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = np.random.random(2 * self.num_of_inputs)
        self.errors = []  # wartosci funkcji kosztu +  jej wykres

    def train(self, training_x, training_y):
        for _ in range(self.iterations):
            e = 0
            for x, y in zip(training_x, training_y):  # losowosc przykladow
                out = self.output(x)
                x = np.array(x)
                self.weights += self.learning_rate * (y - out) * x
                e[y] += (y - out) ** 2
            self.errors.append(e)

    def _activation(self, x):
        return x

    def output(self, x):
        inp = np.concatenate([x, fourier(x)])
        return self._activation(np.dot(self.weights, inp))
