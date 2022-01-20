import matplotlib.pyplot as plt
import numpy as np


X = np.array([0.0, 2.0, 5.0, 11.0])
Y = np.array([2.1, 4.0, 5.5, 8.9])

W = np.array([0.1, 0.1, 0.1])
b = np.array([0.1, 0.1, 0.1])

W1 = np.zeros((1, 8))
b1 = np.zeros((8, ))
W2 = np.zeros((8, 1))
b2 = np.zeros((1, ))


def linear(W, b, x):
    prod_W = np.squeeze(W.T @ np.expand_dims(x, axis=-1), axis=-1)
    return W * x + b


def tanh(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


def d_tanh(a):
    return (4 * np.exp(2 * a)) / (np.exp(2 * a) + 1) ** 2


def model(W, b, x):
    layer_1 = linear(W[0], b[0], x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W[1], b[1], layer_2)
    layer_4 = tanh(layer_3)
    layer_5 = linear(W[2], b[2], layer_4)
    return layer_5

def dy_loss(y, y_prim):
    return np.mean(-2*(y-y_prim))

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W


def f_dW1_loss(y, y_prim, W, b, x):
    layer_1 = dx_linear(W[2], b[2], tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x)))))
    layer_2 = layer_1 * d_tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x))))
    layer_3 = layer_2 * dx_linear(W[1], b[1], tanh(linear(W[0], b[0], x)))
    layer_4 = layer_3 * d_tanh(linear(W[0], b[0], x))
    layer_5 = layer_4 * dW_linear(W[0], b[0], x)
    return np.mean(layer_5) * dy_loss(y, y_prim)

def f_dW2_loss(y, y_prim, W, b, x):
    layer_1 = dx_linear(W[2], b[2], tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x)))))
    layer_2 = layer_1 * d_tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x))))
    layer_3 = layer_2 * dW_linear(W[1], b[1], tanh(linear(W[0], b[0], x)))
    return np.mean(layer_3) * dy_loss(y, y_prim)

def f_dW3_loss(y, y_prim, W, b, x):
    layer_1 = dW_linear(W[2], b[2], tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x)))))
    return np.mean(layer_1) * dy_loss(y, y_prim)



def f_db1_loss(y, y_prim, W, b, x):
    result = d_tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x))))
    result *= tanh(linear(W[0], b[0], x))
    return np.mean(result) * dy_loss(y, y_prim)

def f_db2_loss(y, y_prim, W, b, x):
    return np.mean(d_tanh(linear(W[1], b[1], tanh(linear(W[0], b[0], x))))) * dy_loss(y, y_prim)

def f_db3_loss(y, y_prim, W, b, x):
    return dy_loss(y, y_prim)


def f_loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)


learning_rate = 1e-2

losses = []

for epoch in range(1000):
    Y_prim = model(W, b, X)
    loss = f_loss(Y, Y_prim)

    dW1_loss = f_dW1_loss(Y, Y_prim, W, b, X)
    dW2_loss = f_dW2_loss(Y, Y_prim, W, b, X)
    dW3_loss = f_dW3_loss(Y, Y_prim, W, b, X)
    db1_loss = f_db1_loss(Y, Y_prim, W, b, X)
    db2_loss = f_db2_loss(Y, Y_prim, W, b, X)
    db3_loss = f_db3_loss(Y, Y_prim, W, b, X)

    W[0] -= dW1_loss * learning_rate
    W[1] -= dW2_loss * learning_rate
    W[2] -= dW3_loss * learning_rate
    b[0] -= db1_loss * learning_rate
    b[1] -= db2_loss * learning_rate
    b[2] -= db3_loss * learning_rate
    print(f'W: {W}')
    print(f'b: {b}')
    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()
