import os
import pickle
import time
import matplotlib
import sys
import numpy as np
import torch.nn
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 14) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-1
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7

class Dataset(torch.utils.data.Dataset):
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
            X, self.Y, self.labels = pickle.load(fp)


        X = np.array(X)
        self.X_classes = np.array(X[:, :4])

        self.X = np.array(X[:, 4:]).astype(np.float32)
        X_max = np.max(self.X, axis=0)
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - (X_max + X_min) * 0.5) / ((X_max - X_min) * 0.5)

        self.Y = np.array(self.Y).astype(np.float32)
        Y_max = np.max(self.Y, axis=0)
        Y_min = np.min(self.Y, axis=0)
        self.Y = (self.Y - (Y_max + Y_min) * 0.5) / ((Y_max - Y_min) * 0.5)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.X_classes[idx]), np.array(self.Y[idx])

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

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
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        batch = self.dataset[idx_start:idx_end]
        self.idx_batch += 1
        return batch



dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W'= W - dW * alpha
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 + 3 * 4, out_features=8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=4, out_features=1)
        )
        self.embs = torch.nn.ModuleList()
        for i in range(4):
            self.embs.append(
                torch.nn.Embedding(
                    num_embeddings = len(dataset_full.labels[i]),
                    embedding_dim = 3
                )
            )

    def forward(self, x, x_classes):

        x_emb_list = []
        for i, emb in enumerate(self.embs):
            x_emb_list.append(
                emb.forward(x_classes[:, i])
            )

        x_emb = torch.cat(x_emb_list, dim=-1)
        x_cat = torch.cat([x, x_emb], dim=-1)

        y_prim = self.layers.forward(x_cat)
        return y_prim


class HuberLoss(torch.nn.Module):

    def __init__(self, delta):
        super().__init__()
        self.delta = delta


    def forward(self, y_prim, y):
        return torch.mean(self.delta**2 * (torch.sqrt(1 + ((y - y_prim) / self.delta)**2) - 1))

    def backward(self, y_prim, y):
        #return torch.mean(self.delta**2 * (torch.sqrt(1 + ((y - y_prim) / self.delta)**2) - 1)) - torch.mean(self.delta**2 * torch.mean((y - y_prim) / torch.sqrt(((y - y_prim) / self.delta)**2 + 1)))
        return torch.mean(self.delta**2 * torch.mean((y - y_prim) / torch.sqrt(((y - y_prim) / self.delta)**2 + 1)))

model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = HuberLoss(delta=0.5)


loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, x_classes, y in dataloader:

            y_prim = model.forward(x, x_classes)
            loss = loss_fn.forward(y_prim, y)

            losses.append(loss.item())


            if dataloader == dataloader_train:
                loss = loss_fn.backward(y_prim, y)
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')
    if epoch % 20 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
    # print(
    #     f'epoch: {epoch} '
    #     f'loss_train: {loss_plot_train[-1]} '
    #     f'loss_test: {loss_plot_test[-1]}'
    #     f'acc_train: {acc_plot_train[-1]} '
    #     f'acc_test: {acc_plot_test[-1]}'
    # )

    # if epoch % 10 == 0:
    #     _, axes = plt.subplots(nrows=2, ncols=1)
    #     ax1 = axes[0]
    #     #ax1.title("Loss")
    #     ax1.plot(loss_plot_train, 'r-', label='train')
    #     ax2 = ax1.twinx()
    #     ax2.plot(loss_plot_test, 'c-', label='test')
    #     ax1.legend()
    #     ax2.legend(loc='upper left')
    #     ax1.set_xlabel("Epoch")
    #     ax1.set_ylabel("Loss")
    #
    #     ax1 = axes[1]
    #     #ax1.title("Acc")
    #     ax1.plot(acc_plot_train, 'r-', label='train')
    #     ax2 = ax1.twinx()
    #     ax2.plot(acc_plot_test, 'c-', label='test')
    #     ax1.legend()
    #     ax2.legend(loc='upper left')
    #     ax1.set_xlabel("Epoch")
    #     ax1.set_ylabel("Acc.")
    #     plt.show()