# coding: utf-8
"""
All-in-one CuPy version of the Deep Learning from Scratch style CNN.

- No PyTorch.
- Neural-network computations use CuPy on the GPU.
- The book's dataset.mnist.load_mnist is only used to load MNIST on the CPU;
  the whole dataset is then copied once to GPU memory with cp.asarray().

Place this file in a chapter directory such as ch08 so that ../dataset/mnist.py
can be imported, or adjust sys.path as needed.
"""

import os
import sys
import time
import pickle
import itertools
import cupy as cp

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataset.mnist import load_mnist


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn" / "cupy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility: im2col / col2im
# ============================================================

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Convert 4-D image batches to a 2-D matrix for convolution."""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = cp.pad(
        input_data,
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        mode="constant",
    )

    col = cp.zeros(
        (N, C, filter_h, filter_w, out_h, out_w),
        dtype=input_data.dtype,
    )

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[
                :, :, y:y_max:stride, x:x_max:stride
            ]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Inverse operation of im2col used in convolution/pooling backward."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = cp.zeros(
        (
            N,
            C,
            H + 2 * pad + stride - 1,
            W + 2 * pad + stride - 1,
        ),
        dtype=col.dtype,
    )

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# ============================================================
# Functions
# ============================================================

def identity_function(x):
    return x


def step_function(x):
    return cp.array(x > 0, dtype=cp.int32)


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return cp.maximum(0, x)


def relu_grad(x):
    grad = cp.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    x = x - cp.max(x, axis=-1, keepdims=True)
    exp_x = cp.exp(x)
    return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * cp.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # If labels are one-hot vectors, convert to class indices.
    if t.size == y.size:
        t = t.argmax(axis=1)

    # Ensure class-index labels can be used for indexing.
    t = t.astype(cp.int64, copy=False)

    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)


# ============================================================
# Numerical gradient (debugging only)
# ============================================================

def _numerical_gradient_1d(f, x):
    # This intentionally mirrors the book. It is very slow on a GPU because
    # it updates one element at a time; use it only for gradient checking.
    h = 1e-4
    grad = cp.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)

    grad = cp.zeros_like(X)
    for idx in range(X.shape[0]):
        grad[idx] = _numerical_gradient_1d(f, X[idx])
    return grad


def numerical_gradient(f, x):
    h = 1e-4
    grad = cp.zeros_like(x)

    ranges = [range(size) for size in x.shape]
    for idx in itertools.product(*ranges):
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# ============================================================
# Layers
# ============================================================

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # Copy avoids modifying the caller's gradient buffer in-place.
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return cp.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            labels = self.t.astype(cp.int64, copy=False)
            dx[cp.arange(batch_size), labels] -= 1
            dx = dx / batch_size

        return dx * dout


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(
        self,
        gamma,
        beta,
        momentum=0.9,
        running_mean=None,
        running_var=None,
    ):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            _, D = x.shape
            self.running_mean = cp.zeros(D, dtype=x.dtype)
            self.running_var = cp.zeros(D, dtype=x.dtype)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = cp.mean(xc ** 2, axis=0)
            std = cp.sqrt(var + 1e-6)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / cp.sqrt(self.running_var + 1e-6)

        return self.gamma * xn + self.beta

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = cp.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -cp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = cp.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = 1 + (H + 2 * self.pad - FH) // self.stride
        out_w = 1 + (W + 2 * self.pad - FW) // self.stride

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = cp.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = cp.dot(dout, self.col_W.T)
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = 1 + (H + 2 * self.pad - self.pool_h) // self.stride
        out_w = 1 + (W + 2 * self.pad - self.pool_w) // self.stride

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(
            dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1
        )

        return col2im(
            dcol,
            self.x.shape,
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        )


# ============================================================
# Optimizers
# ============================================================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: cp.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: cp.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: cp.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (cp.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: cp.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (cp.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: cp.zeros_like(val) for key, val in params.items()}
            self.v = {key: cp.zeros_like(val) for key, val in params.items()}

        self.iter += 1
        lr_t = self.lr * cp.sqrt(1.0 - self.beta2 ** self.iter) / (
            1.0 - self.beta1 ** self.iter
        )

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (
                grads[key] * grads[key] - self.v[key]
            )
            params[key] -= lr_t * self.m[key] / (cp.sqrt(self.v[key]) + 1e-7)


# ============================================================
# MyConvNet
# ============================================================

class MyConvNet:
    """
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool -
    affine - relu - dropout - affine - softmax
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_0={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_1={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_2={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_3={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        hidden_size=50,
        output_size=10,
    ):
        pre_node_nums = cp.array(
            [
                1 * 3 * 3,
                32 * 3 * 3,
                32 * 3 * 3,
                64 * 3 * 3,
                64 * 7 * 7,
                hidden_size,
            ],
            dtype=cp.float32,
        )

        weight_init_scales = cp.sqrt(2.0 / pre_node_nums)

        self.params = {}

        self.params["W1"] = (
            weight_init_scales[0] * cp.random.randn(32, 1, 3, 3)
        ).astype(cp.float32)
        self.params["b1"] = cp.zeros(32, dtype=cp.float32)

        self.params["W2"] = (
            weight_init_scales[1] * cp.random.randn(32, 32, 3, 3)
        ).astype(cp.float32)
        self.params["b2"] = cp.zeros(32, dtype=cp.float32)

        self.params["W3"] = (
            weight_init_scales[2] * cp.random.randn(64, 32, 3, 3)
        ).astype(cp.float32)
        self.params["b3"] = cp.zeros(64, dtype=cp.float32)

        self.params["W4"] = (
            weight_init_scales[3] * cp.random.randn(64, 64, 3, 3)
        ).astype(cp.float32)
        self.params["b4"] = cp.zeros(64, dtype=cp.float32)

        self.params["W5"] = (
            weight_init_scales[4] * cp.random.randn(64 * 7 * 7, hidden_size)
        ).astype(cp.float32)
        self.params["b5"] = cp.zeros(hidden_size, dtype=cp.float32)

        self.params["W6"] = (
            weight_init_scales[5] * cp.random.randn(hidden_size, output_size)
        ).astype(cp.float32)
        self.params["b6"] = cp.zeros(output_size, dtype=cp.float32)

        self.layers = []
        self.layers.append(
            Convolution(
                self.params["W1"],
                self.params["b1"],
                conv_param_0["stride"],
                conv_param_0["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(
            Convolution(
                self.params["W2"],
                self.params["b2"],
                conv_param_1["stride"],
                conv_param_1["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(
            Convolution(
                self.params["W3"],
                self.params["b3"],
                conv_param_2["stride"],
                conv_param_2["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(
            Convolution(
                self.params["W4"],
                self.params["b4"],
                conv_param_3["stride"],
                conv_param_3["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params["W5"], self.params["b5"]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params["W6"], self.params["b6"]))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = cp.argmax(t, axis=1)

        acc = cp.array(0, dtype=cp.int64)

        for i in range(0, x.shape[0], batch_size):
            tx = x[i:i + batch_size]
            tt = t[i:i + batch_size]

            y = self.predict(tx, train_flg=False)
            y = cp.argmax(y, axis=1)
            acc += cp.sum(y == tt)

        # One GPU -> CPU scalar transfer at the end of evaluation.
        return (acc / x.shape[0]).item()

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()

        for layer in tmp_layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers[0].dW
        grads["b1"] = self.layers[0].db
        grads["W2"] = self.layers[2].dW
        grads["b2"] = self.layers[2].db
        grads["W3"] = self.layers[5].dW
        grads["b3"] = self.layers[5].db
        grads["W4"] = self.layers[7].dW
        grads["b4"] = self.layers[7].db
        grads["W5"] = self.layers[10].dW
        grads["b5"] = self.layers[10].db
        grads["W6"] = self.layers[13].dW
        grads["b6"] = self.layers[13].db
        return grads

    def save_params(self, file_name="my_params_cupy.pkl"):
        params = {}
        for key, val in self.params.items():
            # Pickle CPU arrays so the file is portable.
            params[key] = cp.asnumpy(val)

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="my_params_cupy.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = cp.asarray(val, dtype=cp.float32)

        self.layers[0].W = self.params["W1"]
        self.layers[0].b = self.params["b1"]
        self.layers[2].W = self.params["W2"]
        self.layers[2].b = self.params["b2"]
        self.layers[5].W = self.params["W3"]
        self.layers[5].b = self.params["b3"]
        self.layers[7].W = self.params["W4"]
        self.layers[7].b = self.params["b4"]
        self.layers[10].W = self.params["W5"]
        self.layers[10].b = self.params["b5"]
        self.layers[13].W = self.params["W6"]
        self.layers[13].b = self.params["b6"]


# ============================================================
# Trainer
# ============================================================

class Trainer:
    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
        evaluate_sample_num_per_epoch=None,
        verbose=True,
    ):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        optimizer_class_dict = {
            "sgd": SGD,
            "momentum": Momentum,
            "nesterov": Nesterov,
            "adagrad": AdaGrad,
            "rmsprop": RMSprop,
            "adam": Adam,
        }
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = cp.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)

        # Keep the scalar on GPU; converting every batch with .item() would
        # synchronize CUDA and noticeably slow the speed test.
        self.train_loss_list.append(loss)

        if self.verbose:
            print("train loss:" + str(loss.item()))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            if self.evaluate_sample_num_per_epoch is not None:
                sample_num = self.evaluate_sample_num_per_epoch
                x_train_sample = self.x_train[:sample_num]
                t_train_sample = self.t_train[:sample_num]
                x_test_sample = self.x_test[:sample_num]
                t_test_sample = self.t_test[:sample_num]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            print(
                "=== epoch:"
                + str(self.current_epoch)
                + ", train acc:"
                + str(train_acc)
                + ", test acc:"
                + str(test_acc)
                + " ==="
            )

        self.current_iter += 1

    def train(self):
        # CuPy/CUDA operations are asynchronous, so synchronize before/after timing.
        cp.cuda.Stream.null.synchronize()

        start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.perf_counter()
        print("=============== Training Start ===============")
        print("start time:", start_wall)

        for _ in range(self.max_iter):
            self.train_step()

        cp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = end_time - start_time

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        print("=============== Final Test Accuracy ===============")
        print("test acc:" + str(test_acc))
        print("=============== Training Timing ===============")
        print("start time:", start_wall)
        print("end time  :", end_wall)
        print(f"training time: {elapsed:.2f} s")
        print(f"training time: {elapsed / 60:.2f} min")

        # Also save the timing result to disk in case the terminal is closed
        # or you are away when training finishes.
        with open(OUTPUT_DIR / "cupy_training_timing.txt", "w", encoding="utf-8") as f:
            f.write(f"start time: {start_wall}\n")
            f.write(f"end time: {end_wall}\n")
            f.write(f"training time: {elapsed:.2f} s\n")
            f.write(f"training time: {elapsed / 60:.2f} min\n")
            f.write(f"final test accuracy: {test_acc}\n")

        print("Timing saved: cupy_training_timing.txt")


# ============================================================
# Main
# ============================================================

def main():
    program_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    program_start = time.perf_counter()

    print("Program started at:", program_start_wall)
    print("CuPy version:", cp.__version__)
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())

    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = props["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print("GPU:", device_name)

    # Load using the book's CPU loader, then copy the complete MNIST dataset
    # once to GPU memory. No CPU<->GPU copy is done per mini-batch.
    (x_train_np, t_train_np), (x_test_np, t_test_np) = load_mnist(
        normalize=True,
        flatten=False,
        one_hot_label=True,
    )

    x_train = cp.asarray(x_train_np, dtype=cp.float32)
    t_train = cp.asarray(t_train_np, dtype=cp.float32)
    x_test = cp.asarray(x_test_np, dtype=cp.float32)
    t_test = cp.asarray(t_test_np, dtype=cp.float32)

    print("x_train:", x_train.shape, x_train.dtype)
    print("t_train:", t_train.shape, t_train.dtype)
    print("x_test :", x_test.shape, x_test.dtype)
    print("Data has been copied to GPU memory.")

    network = MyConvNet()

    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        # Full train/test accuracy each epoch is expensive and makes timing
        # less comparable to your PyTorch loop. Use 1000 for a faster speed
        # test, or set None for the book's full-dataset evaluation behavior.
        evaluate_sample_num_per_epoch=1000,
        # Printing every mini-batch forces GPU synchronization and is slow.
        verbose=False,
    )

    trainer.train()

    network.save_params(str(OUTPUT_DIR / "my_params_cupy.pkl"))
    print("Saved: my_params_cupy.pkl")

    cp.cuda.Stream.null.synchronize()
    program_end = time.perf_counter()
    program_end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    program_elapsed = program_end - program_start

    print("=============== Whole Program Timing ===============")
    print("program start:", program_start_wall)
    print("program end  :", program_end_wall)
    print(f"whole program time: {program_elapsed:.2f} s")
    print(f"whole program time: {program_elapsed / 60:.2f} min")

    # Append whole-program timing so the result remains available after completion.
    with open(OUTPUT_DIR / "cupy_training_timing.txt", "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"program start: {program_start_wall}\n")
        f.write(f"program end: {program_end_wall}\n")
        f.write(f"whole program time: {program_elapsed:.2f} s\n")
        f.write(f"whole program time: {program_elapsed / 60:.2f} min\n")


if __name__ == "__main__":
    main()
