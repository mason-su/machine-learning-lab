# Create your awesome net!!
# coding: utf-8
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *

class MyConvNet:
    '''MyConvNet architecture:
        conv - relu - conv - relu - pool -
        conv - relu - conv - relu - pool -
        affine - relu -dropout - affine - softmax
    '''
    def __init__(self, input_dim = (1,28, 28),
                 conv_param_0 = {'filter_num': 32, 'filter_size': 3, 'pad':1, 'stride': 1},
                 conv_param_1 = {'filter_num': 32, 'filter_size': 3, 'pad':1, 'stride': 1},
                 conv_param_2 = {'filter_num': 64, 'filter_size': 3, 'pad':1, 'stride': 1},
                 conv_param_3 = {'filter_num': 64, 'filter_size': 3, 'pad':1, 'stride': 1},
                 hidden_size = 50, output_size = 10):
       # Weight initialization ===========
               # Number of connections each neuron in each layer has
               # with neurons in the previous layer (TODO: calculate automatically)
        pre_node_nums = np.array([
            1*3*3,
            32*3*3,
            32*3*3,
            64*3*3,
            64*7*7,
            hidden_size
        ]) 
        # He Initialization
        weight_init_scales = np.sqrt(2.0 /pre_node_nums)

        self.params = {}

        #conv 1
        self.params['W1'] = weight_init_scales[0] * np.random.randn(
            32, 1, 3, 3
        ) 
        self.params['b1'] = np.zeros(32)

        #conv 2
        self.params['W2'] = weight_init_scales[1] * np.random.randn(
            32, 32, 3, 3
        )
        self.params['b2'] = np.zeros(32)

        #conv 3
        self.params['W3'] = weight_init_scales[2] * np.random.randn(
            64, 32, 3, 3
        )
        self.params['b3'] = np.zeros(64)

        #conv 4
        self.params['W4'] = weight_init_scales[3] * np.random.randn(
            64, 64, 3, 3
        )
        self.params['b4'] = np.zeros(64)

        #affine 1
        self.params["W5"] = weight_init_scales[4] * np.random.randn(
            64*7*7, hidden_size
        )
        self.params['b5'] = np.zeros(hidden_size)

        #affine 2
        self.params['W6'] = weight_init_scales[5] * np.random.randn(
            hidden_size,output_size
        )
        self.params['b6'] = np.zeros(output_size)

        #====================create layers===============

        self.layers = []

        #conv1
        self.layers.append(
            Convolution(
                self.params['W1'],
                self.params['b1'],
                conv_param_0['stride'],
                conv_param_0['pad']
            )
        )
        #Relu 1
        self.layers.append(Relu())
        #conv 2
        self.layers.append(
            Convolution(
                self.params['W2'],
                self.params['b2'],
                conv_param_1['stride'],
                conv_param_1['pad']
            )
        )
        #Relu 2
        self.layers.append(Relu())
        #Pool 1
        self.layers.append(Pooling(pool_h = 2, pool_w = 2, stride = 2))
        #Conv 3
        self.layers.append(
            Convolution(
                self.params['W3'],
                self.params['b3'],
                conv_param_2['stride'],
                conv_param_2['pad']
            )
        )
        #Relu 3
        self.layers.append(Relu())
        #Conv 4
        self.layers.append(
            Convolution(
                self.params['W4'],
                self.params['b4'],
                conv_param_3['stride'],
                conv_param_3['pad']
            )
        )
        #Relu 4
        self.layers.append(Relu())
        #Pool 2
        self.layers.append(
            Pooling(pool_h = 2, pool_w = 2, stride = 2)
        )
        #affine 1
        self.layers.append(
            Affine(
                self.params['W5'],
                self.params['b5']
            )
        )
        #relu 5
        self.layers.append(Relu())
        #dropout1
        self.layers.append(Dropout(0.5))
        #affine 2
        self.layers.append(
            Affine(
                self.params['W6'],
                self.params['b6']
            )
        )
        #Softmax+loss
        self.last_layer = SoftmaxWithLoss()

    #forward propogation
    def predict(self, x, train_flg = False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    #loss
    def loss(self, x, t):
        y = self.predict(x, train_flg = True)
        return self.last_layer.forward(y,t)

    #accuracy
    def accuracy(self, x, t, batch_size = 100):
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)

        acc = 0.0

        for i in range(0, x.shape[0],batch_size):
            tx = x[i:i+batch_size]
            tt = t[i:i+batch_size]

            y = self.predict(tx, train_flg = False)
            y = np.argmax(y, axis = 1)

            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        #forward propagation
        self.loss(x,t)

        #backward propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()

        for layer in tmp_layers:
            dout = layer.backward(dout)

        #collect gradients
        grads = {}
        grads['W1'] = self.layers[0].dW
        grads['b1'] = self.layers[0].db

        grads['W2'] = self.layers[2].dW
        grads['b2'] = self.layers[2].db

        grads['W3'] = self.layers[5].dW
        grads['b3'] = self.layers[5].db

        grads['W4'] = self.layers[7].dW
        grads['b4'] = self.layers[7].db

        grads['W5'] = self.layers[10].dW
        grads['b5'] = self.layers[10].db

        grads['W6'] = self.layers[13].dW
        grads['b6'] = self.layers[13].db

        return grads
    def save_params(self, file_name="my_params.pkl"):
        params = {}

        for key, val in self.params.items():
            params[key] = val

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


    def load_params(self, file_name="my_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # Load parameters into self.params
        for key, val in params.items():
            self.params[key] = val

        # Update the parameters stored inside each layer
        self.layers[0].W = self.params['W1']
        self.layers[0].b = self.params['b1']

        self.layers[2].W = self.params['W2']
        self.layers[2].b = self.params['b2']

        self.layers[5].W = self.params['W3']
        self.layers[5].b = self.params['b3']

        self.layers[7].W = self.params['W4']
        self.layers[7].b = self.params['b4']

        self.layers[10].W = self.params['W5']
        self.layers[10].b = self.params['b5']

        self.layers[13].W = self.params['W6']
        self.layers[13].b = self.params['b6']

        

    

    



