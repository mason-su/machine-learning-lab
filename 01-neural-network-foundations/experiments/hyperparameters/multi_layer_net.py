# coding: utf-8

import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class MultiLayerNet:

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        weight_init_std=0.01
    ):

        # ====================================================
        # hidden_sizes 必须是一个 list
        #
        # 例如：
        # [100]
        # [100, 100]
        # [100, 50, 20]
        # ====================================================

        self.hidden_sizes = hidden_sizes

        # ====================================================
        # layer_sizes 保存每一层的神经元数量
        #
        # 例如：
        #
        # input_size = 784
        # hidden_sizes = [100, 50]
        # output_size = 10
        #
        # layer_sizes =
        # [784, 100, 50, 10]
        # ====================================================

        self.layer_sizes = (
            [input_size]
            + hidden_sizes
            + [output_size]
        )


        # ====================================================
        # Affine 层总数
        #
        # [784,100,50,10]
        #
        # 有：
        #
        # Affine1: 784 → 100
        # Affine2: 100 → 50
        # Affine3: 50 → 10
        #
        # 所以 affine_layer_count = 3
        # ====================================================

        self.affine_layer_count = (
            len(self.layer_sizes) - 1
        )


        # ====================================================
        # 1. 权重初始化
        # ====================================================

        self.params = {}


        for layer_index in range(
            1,
            self.affine_layer_count + 1
        ):

            previous_size = (
                self.layer_sizes[layer_index - 1]
            )

            current_size = (
                self.layer_sizes[layer_index]
            )


            self.params[
                'W' + str(layer_index)
            ] = (

                weight_init_std
                * np.random.randn(
                    previous_size,
                    current_size
                )
            )


            self.params[
                'b' + str(layer_index)
            ] = np.zeros(
                current_size
            )


        # ====================================================
        # 2. 自动生成网络 Layer
        # ====================================================

        self.layers = OrderedDict()


        for layer_index in range(
            1,
            self.affine_layer_count + 1
        ):

            # -----------------------------------------------
            # Affine Layer
            # -----------------------------------------------

            self.layers[
                'Affine' + str(layer_index)
            ] = Affine(

                self.params[
                    'W' + str(layer_index)
                ],

                self.params[
                    'b' + str(layer_index)
                ]
            )


            # -----------------------------------------------
            # 最后一层 Affine 后面不要 ReLU
            #
            # Hidden layers:
            #
            # Affine → ReLU
            #
            # Output layer:
            #
            # Affine → SoftmaxWithLoss
            # -----------------------------------------------

            if layer_index < self.affine_layer_count:

                self.layers[
                    'Relu' + str(layer_index)
                ] = Relu()


        # ====================================================
        # 输出层
        # ====================================================

        self.lastLayer = SoftmaxWithLoss()


    # ========================================================
    # Forward
    # ========================================================

    def predict(self, x):

        for layer in self.layers.values():

            x = layer.forward(x)

        return x


    # ========================================================
    # Loss
    # ========================================================

    def loss(self, x, t):

        y = self.predict(x)

        return self.lastLayer.forward(
            y,
            t
        )


    # ========================================================
    # Accuracy
    # ========================================================

    def accuracy(self, x, t):

        y = self.predict(x)

        y = np.argmax(
            y,
            axis=1
        )


        if t.ndim != 1:

            t = np.argmax(
                t,
                axis=1
            )


        accuracy = (

            np.sum(y == t)

            / float(x.shape[0])
        )

        return accuracy


    # ========================================================
    # Numerical Gradient
    # ========================================================

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t)

        grads = {}


        for layer_index in range(
            1,
            self.affine_layer_count + 1
        ):

            grads[
                'W' + str(layer_index)
            ] = numerical_gradient(

                loss_W,

                self.params[
                    'W' + str(layer_index)
                ]
            )


            grads[
                'b' + str(layer_index)
            ] = numerical_gradient(

                loss_W,

                self.params[
                    'b' + str(layer_index)
                ]
            )


        return grads


    # ========================================================
    # Backpropagation Gradient
    # ========================================================

    def gradient(self, x, t):

        # ====================================================
        # Forward
        # ====================================================

        self.loss(x, t)


        # ====================================================
        # Backward
        # ====================================================

        dout = 1

        dout = self.lastLayer.backward(
            dout
        )


        layers = list(
            self.layers.values()
        )

        layers.reverse()


        for layer in layers:

            dout = layer.backward(
                dout
            )


        # ====================================================
        # 保存 Gradient
        # ====================================================

        grads = {}


        for layer_index in range(
            1,
            self.affine_layer_count + 1
        ):

            grads[
                'W' + str(layer_index)
            ] = self.layers[
                'Affine' + str(layer_index)
            ].dW


            grads[
                'b' + str(layer_index)
            ] = self.layers[
                'Affine' + str(layer_index)
            ].db


        return grads