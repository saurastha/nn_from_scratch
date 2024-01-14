import numpy as np


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.maximum(0, x)


class Softmax:

    def __init__(self):
        self.output = None

    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
