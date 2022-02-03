import matplotlib.pyplot as plt
import numpy as np


X = np.array([[0.0, 1.0], [2.0, 1.0], [5.0, 1.0], [11.0, 1.0], [16.0, 1.0]])
Y = np.array([2.1, 4.0, 5.5, 8.9, 13.1])
X = np.expand_dims(X, -1)
W_old = np.array([0.1, 0.1, 0.1])
b_old = np.array([0.1, 0.1, 0.1])

W1 = np.zeros((1, 3))
b1 = np.zeros((3,))
W2 = np.zeros((3, 1))
b2 = np.zeros((1, ))
W3 = np.zeros((1, 3))
b3 = np.zeros((3, ))



def linear(W, b, x):

    #print("----linear----")
    prod_W = np.squeeze(W.T @ np.expand_dims(x, axis=-1), axis=-1)
    result = prod_W + b
    #print(f'result: {np.shape(prod_W)}')
    #print("------\/------")

    return result


def tanh(a):
    #print("----tanh_a----")
    #print(f'a: {np.shape(a)}')
    result = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
    #print(f'result : {np.shape(result)}')
    #print("------\/------")
    return result


def d_tanh(a):
    return (4 * np.exp(2 * a)) / (np.exp(2 * a) + 1) ** 2


def model(W1, W2, W3, b1, b2, b3, x):
    layer_1 = linear(W1, b1, x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W2, b2, layer_2)
    layer_4 = tanh(layer_3)
    layer_5 = linear(W3, b3, layer_4)
    #print(layer_5.T)
    return layer_5.T

def dy_loss(y, y_prim):
   loss = -2*(y-y_prim)
   return np.array([loss[1]]).T


def dy_loss_2(y, y_prim):
   loss = -2*(y-y_prim)
   return loss

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W


def f_dW1_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    #print("-----------------------------------------")
    layer_1 = np.expand_dims(dx_linear(W3, b3, tanh(linear(W2, b2, tanh(linear(W1, b1, x))))).T, axis=-1)
    #print(f"layer_1: {np.shape(layer_1)}")
    layer_2 = d_tanh(linear(W2, b2, tanh(linear(W1, b1, x))))
    #print(f"layer_2: {np.shape(layer_2)}")
    layer_3 = np.expand_dims(dx_linear(W2, b2, tanh(linear(W1, b1, x))).T, axis=-1)
    #print(f"layer_3: {np.shape(layer_3)}")
    layer_4 = d_tanh(linear(W1, b1, x))
    #print(f"layer_4: {np.shape(layer_4)}")
    loss = dy_loss(y, y_prim)
    #print(f"loss: {np.shape((loss))}")
    layer_5 = dW_linear(W1, b1, x)
    #print(f"layer_5: {np.shape(layer_5)}")
    dot_1 = np.squeeze(loss @ layer_1, axis=-1).T
    #print(f"dot_1: {np.shape(dot_1)}")
    res = dot_1
    res = res * layer_2
    #print(f"res_2: {np.shape(res)}")
    res = np.squeeze(res @ layer_3, axis=-1).T
    #print(f"res_3: {np.shape(res)}")
    res = res * layer_4
    #print(f"res_4: {np.shape(res)}")
    res = res.T @ layer_5
    #print(f"res_5.T: {np.shape(res.T)}")
    #print(res.T)
    return res.T

def f_dW2_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    #print("-----------------f_dW2_loss----------------------")
    layer_3 = dW_linear(W2, b2, tanh(linear(W1, b1, x)))
    layer_2 = d_tanh(linear(W2, b2, tanh(linear(W1, b1, x))))
    layer_1 = np.expand_dims(dx_linear(W3, b3, tanh(linear(W2, b2, tanh(linear(W1, b1, x))))).T, axis=-1)
    #print(f"layer_1: {np.shape(layer_1)}")
    #print(f"layer_2: {np.shape(layer_2)}")
    #print(f"layer_3: {np.shape(layer_3)}")
    loss = dy_loss(y, y_prim)
    #print(f"loss: {np.shape((loss))}")
    dot_1 = np.squeeze(loss @ layer_1, axis=-1).T
    print(f"dot_1: {np.shape(dot_1)}")
    res = dot_1 * layer_3
    #print(f"res_2: {np.shape(res)}")
    res = res.T @ layer_2
    #print(f"res_3: {np.shape(res)}")
    #print(res)
    return res

def f_dW3_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    #print("-----------------f_dW3_loss----------------------")
    layer_1 = dW_linear(W3, b3, tanh(linear(W2, b2, tanh(linear(W1, b1, x)))))
    #print(f"layer_1: {np.shape(layer_1)}")
    loss = dy_loss_2(y, y_prim)
    #print(f"loss: {np.shape((loss))}")
    res = loss @ layer_1
    #print(f"res: {np.shape(res.T)}")
    return res.T



def f_db1_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    print("-----------------f_db1_loss----------------------")
    layer_1 = d_tanh(linear(W2, b2, tanh(linear(W1, b1, x)))).T
    print(f"layer_1: {np.shape(layer_1)}")
    layer_2 = tanh(linear(W1, b1, x))
    print(f"layer_2: {np.shape(layer_2)}")
    loss = dy_loss(y, y_prim)
    print(f"loss: {np.shape(loss)}")
    res = loss @ layer_1
    print(f"res: {np.shape(res)}")
    res = res @ layer_2
    print(f"res: {np.shape(res.T)}")
    return res.T

def f_db2_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    print("-----------------f_db2_loss----------------------")
    layer_1 = d_tanh(linear(W2, b2, tanh(linear(W1, b1, x)))).T
    print(f"layer_1: {np.shape(layer_1)}")
    loss = dy_loss(y, y_prim)
    print(f"loss: {np.shape(loss)}")
    res = loss @ layer_1
    print(f"res: {np.shape(res)}")
    return res

def f_db3_loss(y, y_prim, W1, W2, W3, b1, b2, b3, x):
    return dy_loss(y, y_prim)


def f_loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)


learning_rate = 1e-2

losses = []

for epoch in range(1000):
    Y_prim = model(W1, W2, W3, b1, b2, b3, X)
    loss = f_loss(Y, Y_prim)

    dW1_loss = f_dW1_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)
    dW2_loss = f_dW2_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)
    dW3_loss = f_dW3_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)
    db1_loss = f_db1_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)
    db2_loss = f_db2_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)
    db3_loss = f_db3_loss(Y, Y_prim, W1, W2, W3, b1, b2, b3, X)

    W1 -= dW1_loss * learning_rate
    print("----------")
    print(W2)
    print("---")
    print(dW2_loss)
    print("----------")
    W2 -= dW2_loss * learning_rate
    W3 -= dW3_loss * learning_rate
    b1 -= db1_loss * learning_rate
    b2 -= db2_loss * learning_rate
    b3 -= db3_loss * learning_rate
    #print(f'W: {W}')
    #print(f'b: {b}')
    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()
