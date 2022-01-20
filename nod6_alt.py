import numpy as np


X = np.array([0.0, 2.0, 5.0, 11.0])
Y = np.array([2.1, 4.0, 5.5, 8.9])

W1=0.01
W2=0.01
W3=0.01
b1=0.01
b2=0.01
b3=0.01

learning_rate = 0.1


def linear(W, b, x):
    return W * x + b


def tanh(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


def d_tanh(a):
    return (4 * np.exp(2 * a)) / (np.exp(2 * a) + 1) ** 2


def model(W1, W2, W3, b1, b2, b3, x):
    layer_1 = linear(W1, b1, x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W2, b2, layer_2)
    layer_4 = tanh(layer_3)
    layer_5 = linear(W3, b3, layer_4)
    return layer_5

def dy_model(y, y_prim):
    return np.mean(-2*(y-y_prim))

def dW1_model(W1, W2, W3, b1, b2 , b3, x):
    result = W3
    result *= d_tanh(linear(W2, b2, tanh(linear(W1, b1, x))))
    result *= W2
    result *= d_tanh(linear(W1, b1, x))
    result *= x
    return result

def dW2_model(y, y_prim, W1, W2, W3, b1, b2 , b3, x):
    result = W3
    result *= d_tanh(linear(W2, b2, tanh(linear(W1, b1, x))))
    result *= tanh(linear(W1, b1, x))
    return result * dy_model(y, y_prim)

def dW3_model(y, y_prim, W1, W2, W3, b1, b2 , b3, x):
    layer_1 = linear(W1, b1, x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W2, b2, layer_2)
    layer_4 = tanh(layer_3)
    return layer_4 * dy_model(y, y_prim)

def db1_model(y, y_prim, W1, W2, W3, b1, b2 , b3, x):
    result = d_tanh(linear(W2, b2, tanh(linear(W1, b1, x))))
    result *= tanh(linear(W1, b1, x))
    return result * dy_model(y, y_prim)

def db2_model(y, y_prim, W1, W2, W3, b1, b2 , b3, x):
    return d_tanh(linear(W2, b2, tanh(linear(W1, b1, x)))) * dy_model(y, y_prim)

def db3_model(y, y_prim, W1, W2, W3, b1, b2 , b3, x):
    return 1 * dy_model(y, y_prim)


def f_loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)


for epoch in range(100):
    Y_prim = model(W1, W2, W3, b1, b2, b3, X)
    loss = f_loss(Y, Y_prim)

    dW1_loss = dW1_model(W1, W2, W3, b1, b2, b3, X)
    dW2_loss = dW2_model(Y, Y_prim, W1, W2, W3, b1, b2 , b3, X)
    dW3_loss = dW3_model(Y, Y_prim, W1, W2, W3, b1, b2 , b3, X)
    db1_loss = db1_model(Y, Y_prim, W1, W2, W3, b1, b2 , b3, X)
    db2_loss = db2_model(Y, Y_prim, W1, W2, W3, b1, b2 , b3, X)
    db3_loss = db3_model(Y, Y_prim, W1, W2, W3, b1, b2 , b3, X)

    W1 -= dW1_loss[0] * learning_rate
    W2 -= dW2_loss[0] * learning_rate
    W3 -= dW3_loss[0] * learning_rate
    b1 -= db1_loss[0] * learning_rate
    b2 -= db2_loss[0] * learning_rate
    b3 -= db3_loss * learning_rate
    print(f'W1: {W1}')
    print(f'W2: {W2}')
    print(f'W3: {W3}')
    print(f'b1: {b1}')
    print(f'b2: {b2}')
    print(f'b3: {b3}')
    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss}')