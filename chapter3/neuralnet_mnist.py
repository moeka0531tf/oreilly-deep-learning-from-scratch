# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def getdata():
    # (訓練画像、訓練ラベル), (テスト画像, テストラベル)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sigmoid(a3)
    y = identity_function(z3)
    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



x, t = getdata()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if(p == t[i]):
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt / len(x))))


# バッチ処理
batch_size = 100 # バッチの数
accuracy_cnt_batch = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt_batch += np.sum(p == t[i:i+batch_size])

print("Accuracy_batch:" + str(float(accuracy_cnt / len(x))))