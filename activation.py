import numpy as np


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.maximum(0, x)