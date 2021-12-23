import numpy as np

X = np.array([1, 2, 3, 5])
Y = np.array([0.7, 1.5, 4.5, 9.5])

W = 0
b = 0

def linear(W, b, x):
    return W * x + b

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def da_sigmoid(a):
    return np.exp(-a)/(2 * (np.exp(-a) + 1))

def model(W, b, x):
    return sigmoid(linear(W, b, x)) * 20.0

def dW_model(W, b, x):
    return sigmoid(dW_linear(W, b, x)) * 20.0

def db_model(W, b, x):
    return sigmoid(db_linear(W, b, x)) * 20.0

def f_loss(y, y_prim):
    return np.mean((y - y_prim) ** 2)

def f_dW_loss(y, y_prim, W, b, x): # derivative WRT Loss function
    return -2*(y - y_prim) * (x * np.exp(-(W * x + b))) / (np.exp(-(W * x + b)) + 1)**2

def f_db_loss(y, y_prim, W, b, x):
    return -2*(y - y_prim) * np.exp(-(W * x + b)) / (np.exp(-(W * x + b)) + 1)**2


learning_rate = 2e-2
for epoch in range(1000):

    Y_prim = model(W, b, X)
    loss = f_loss(Y, Y_prim)

    dW_loss = f_dW_loss(Y, Y_prim, W, b, X)
    db_loss = f_db_loss(Y, Y_prim, W, b, X)

    W -= dW_loss * learning_rate
    b -= db_loss * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'loss: {loss}')