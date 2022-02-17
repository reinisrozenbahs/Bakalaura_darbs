import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-2
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7

def normalize(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    return (x - ((x_max + x_min) / 2)) / ((x_max - x_min) / 2)

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        self.X = np.array(self.X)
        self.X = normalize(self.X)

        self.Y = np.array(self.Y)
        self.Y = normalize(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), self.Y[idx]

class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if len(self) < self.idx_batch:
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        batch = self.dataset[idx_start:idx_end]
        X, Y = batch
        Y = np.expand_dims(Y, axis=-1)
        self.idx_batch += 1
        return X, Y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        value_W = np.random.random(size=(in_features, out_features))
        value_W -= 0.5
        value_W *= 2.0
        self.W =  Variable(
            value = value_W,
            grad = np.zeros((BATCH_SIZE, in_features, out_features))
        )
        self.b = Variable(
            value = np.zeros((out_features,)),
            grad = np.zeros((BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        x_3d = np.expand_dims(x.value, axis=-1)
        Wx_3d = self.W.value.T @ x_3d
        Wx_2d = np.squeeze(Wx_3d, axis=-1)
        self.output = Variable(
            Wx_2d + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)

#x_dummy = Variable(value=np.random.random(size=(BATCH_SIZE, 7)))
#linear_dummy = LayerLinear(in_features=7, out_features=2)
#y_dummy = linear_dummy.forward(x_dummy)
#linear_dummy.backward()
#exit()

class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            1.0 / (1.0 + np.exp(-x.value))
        )
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1 - self.output.value) * self.output.grad

class LayerSwish():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            x.value / (1.0 + np.exp(-x.value))
        )
        return self.output

    def backward(self):

        self.x.grad += (self.output.value + (1.0 / (1.0 + np.exp(-self.x.value))) * (1 - self.output.value)) * self.output.grad

class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        x_out = np.array(x.value)

        x_out[x_out < 0] = 0

        self.output = Variable(
            x_out
        )
        return self.output

    def backward(self):
        for elem in self.output.value:
            for elem_2 in elem:
                if elem_2 > 0:
                    self.x.grad += 1 * self.output.grad


class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean((y.value - y_prim.value)**2)
        return loss

    def backward(self):
        self.y_prim.grad += -2 * (self.y.value - y_prim.value)


class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad += -(self.y.value - self.y_prim.value)/ np.abs(self.y.value - self.y_prim.value)

# class LossHuber():
#     def __init__(self, sigma):
#         self.y = None
#         self.y_prim = None
#         self.sigma = sigma
#
#     def forward(self, y: Variable, y_prim: Variable):
#         self.y = y
#         self.y_prim = y_prim
#         loss = np.mean(np.abs(y.value - y_prim.value))
#         return loss
#
#     def backward(self):
#         self.y_prim.grad += -(self.y.value - self.y_prim.value)/ np.abs(self.y.value - self.y_prim.value)

class HuberLoss():

    def __init__(self, delta):
        super().__init__()
        self.y = None
        self.y_prim = None
        self.delta = delta


    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        return np.mean(self.delta**2 * (np.sqrt(1 + ((y.value - y_prim.value) / self.delta)**2) - 1))

    def backward(self):
        self.y_prim.grad += -self.delta**2 * ((self.y.value - self.y_prim.value) / (np.sqrt(1 + ((self.y.value - self.y_prim.value) / self.delta)**2) - 1))

def f_nmrse(y, y_prim):
    mrse_val = (np.mean(np.square(y_prim.value - y.value))) ** (1 / 2)
    stdev_val = np.std(y.value)
    return mrse_val / stdev_val

def f_r_square(y, y_prim):
    res_val = np.sum(np.square(y_prim.value - y.value))
    tot_val = np.sum(np.square(y.value - np.mean(y.value)))
    return 1 - (res_val / tot_val)

class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=7, out_features=4),
            #LayerSigmoid(),
            #LayerReLU(),
            LayerSwish(),
            LayerLinear(in_features=4, out_features=4),
            #LayerSigmoid(),
            #LayerReLU(),
            LayerSwish(),
            LayerLinear(in_features=4, out_features=1)
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables



class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = HuberLoss(delta=0.5)

loss_plot_train = []
loss_plot_test = []
error_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, y in dataloader:

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)
            error_plot_test.append(np.mean(f_r_square(Variable(value=y), y_prim)))
            losses.append(loss)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')
    if epoch % 50 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()


