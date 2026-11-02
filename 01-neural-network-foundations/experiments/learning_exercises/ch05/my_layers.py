# coding: utf-8
import numpy as np
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.functions import *
from common.util import im2col, col2im

class Affine:
    def __init__(self, W, b):
        pass
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.original_x_shape = x.shape

        x = x.reshape(x.shape[0],-1)
        self.x = x
        out = np.dot(self.x, self.W ) + self.b
        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape)

        return dx

class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        # TODO
        pass
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # TODO
        pass
        dout[self.mask] = 0
        dx = dout.copy()
        return dx 

class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.x = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t)/batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx/batch_size

        return dx*dout
