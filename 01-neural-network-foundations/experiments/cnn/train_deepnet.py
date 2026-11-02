# coding: utf-8
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from awesome_net import MyConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = MyConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
(PROJECT_ROOT / "outputs" / "cnn" / "numpy").mkdir(parents=True, exist_ok=True)
network.save_params(str(PROJECT_ROOT / "outputs" / "cnn" / "numpy" / "my_convnet_params.pkl"))
print("Saved Network Parameters!")
