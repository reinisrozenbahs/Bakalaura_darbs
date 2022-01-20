import numpy as np

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

learning_rate = 1e-2

def linear(W, b, x):
    return W * x + b

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W

def relu(x, a):
    y = np.array(x)
    y[y <= 0] *= a
    return y

def dw_relu(x, a):
    y = np.array(x)
    y[y <= 0] = a
    y[y > 0] = 1
    return y

def tanh(a):
    return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))

def da_tanh(a):
    return (4 * np.exp(2 * a))/(np.exp(2 * a) + 1)**2

def model(W, b, x):
    return relu(linear(W, b, tanh(linear(W, b, x))), learning_rate)

def dW_model(W, b, x):
    return da_tanh(linear(W, b, x)) * x * tanh(linear(W, b, x))

def db_model(W, b, x):
    return da_tanh(linear(W, b, x))

def f_loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def f_dW_loss(y, y_prim, W, b, x): # derivative WRT Loss function
    return (y - y_prim)/np.abs(y - y_prim) * dW_model(W, b, x)

def f_db_loss(y, y_prim, W, b, x):
    return (y - y_prim)/np.abs(y - y_prim) * db_model(W, x, b)



for epoch in range(1000):

    Y_prim = model(W, b, X)
    loss = f_loss(Y, Y_prim)

    dW_loss = f_dW_loss(Y, Y_prim, W, b, X)
    db_loss = f_db_loss(Y, Y_prim, W, b, X)

    W -= dW_loss * learning_rate
    b -= db_loss * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss}')