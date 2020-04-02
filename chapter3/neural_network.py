# coding: utf-8
import numpy as np

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2x3の行列,次のニューロンは3つ・現状のニューロンは2つ
    network['b1'] = np.array([0.1, 0.2, 0.3]) # 次のニューロンの個数と一致
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3x2の行列,次のニューロンは2つ・現状のニューロンは3つ
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    # 恒等関数
    # 値をそのまま返す関数、出力層に用いる
    return x

def foward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    print(x)
    print(W1)
    print(b1)
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sigmoid(a3)
    y = identity_function(z3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = foward(network, x)
print(y)