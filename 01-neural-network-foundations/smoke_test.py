"""Quick CPU-only test; does not download MNIST or train a full model."""
import numpy as np
from experiments.hyperparameters.two_layer_net import TwoLayerNet
from experiments.cnn.awesome_net import MyConvNet

rng = np.random.default_rng(1234)
small = TwoLayerNet(784, 5, 10)
x = rng.normal(size=(2,784)).astype(np.float64)
t = np.eye(10)[[1,3]]
loss = small.loss(x,t)
assert np.isfinite(loss)
grad = small.gradient(x,t)
assert grad['W1'].shape == (784,5)
print('PASS: TwoLayerNet loss and gradients')

cnn = MyConvNet()
x = rng.normal(size=(1,1,28,28)).astype(np.float64)
y = cnn.predict(x, train_flg=False)
assert y.shape == (1,10)
assert np.isfinite(y).all()
print('PASS: MyConvNet NumPy forward pass')
