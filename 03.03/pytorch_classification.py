import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (10, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X_tmp, Y_tmp, self.labels = pickle.load(fp)

        # self.labels[1] = ['Diesel', 'Petrol', 'LPG', 'CNG']

        X_tmp = np.array(X_tmp)
        X_classes = np.array(X_tmp[:, :4])
        #print(X_classes)
        self.Y = X_classes[:, 1]

        self.Y_prob = torch.zeros((len(self.Y), len(self.labels[1])))

        idx_range = range(len(self.Y))
        self.Y_prob[idx_range, self.Y] = 1.0

        #self.X_classes = np.concatenate((X_classes[:, :1], X_classes[:, 2:]), axis=-1)
        self.X_classes = X_classes
        #print(self.X_classes)
        X_tmp = np.array(X_tmp[:, 4:]).astype(np.float32)
        Y_tmp = np.expand_dims(Y_tmp, axis=-1).astype(np.float32)
        self.X = np.concatenate((X_tmp, Y_tmp), axis=-1)
        X_max = np.max(self.X, axis=0) # (7, )
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - (X_max + X_min) * 0.5) / (X_max - X_min) * 0.5

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.X_classes[idx]), self.Y_prob[idx]

dataset_full = Dataset()

train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=4 + 3 * 4, out_features=8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=4, out_features=1)
        )

        self.embs = torch.nn.ModuleList()
        for i in range(4): # brand, fuel, transmission, dealership
            self.embs.append(
                torch.nn.Embedding(
                    num_embeddings=len(dataset_full.labels[i]),
                    embedding_dim=3
                )
            )

    def forward(self, x, X_classes):
        x_emb_list = []
        for i, emb in enumerate(self.embs):
            x_emb_list.append(
                emb.forward(X_classes[:, i])
            )
        x_emb = torch.cat(x_emb_list, dim=-1)
        x_cat = torch.cat([x, x_emb], dim=-1)
        #print(x_cat.size())
        y_prim = self.layers.forward(x_cat)
        return y_prim


class LossCCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        return torch.mean(-y/(y_prim + 1e-8))

model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = LossCCE()

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
f1_plot_train = []
f1_plot_test = []

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        f1s = []

        fp = 1
        tp = 1
        tn = 1
        fn = 1

        for x, x_classes, y in dataloader:

            y_prim = model.forward(x, x_classes)
            loss = loss_fn.forward(y_prim, y)

            losses.append(loss.item())

            y_idx = torch.argmax(y, axis=-1)
            y_prim_idx = torch.argmax(y_prim, axis=-1)
            acc = torch.mean((y_idx == y_prim_idx) * 1.0)
            accs.append(acc)

            f1s = 0 # TODO
            accs.append(f1s)

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        conf_matrix = np.array([
            [tp, fn],
            [fp, tn]
        ])

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            f1_plot_train.append(np.mean(f1s))
            conf_matrix_train = conf_matrix
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            f1_plot_test.append(np.mean(f1s))
            conf_matrix_test = conf_matrix

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 10 == 0 or 1:
        plt.tight_layout(pad=0)
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.tight_layout(pad=5)
        ax1 = axes[0, 0]
        ax1.set_title("Loss")
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[1, 0]
        ax1.set_title("Acc")
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc.")

        ax1 = axes[0, 1]
        ax1.set_title("Train Conf.Mat")
        ax1.imshow(conf_matrix_train, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        ax1.set_xticklabels(['Manual', 'Automatic'])
        ax1.set_yticklabels(['Manual', 'Automatic'])
        ax1.set_yticks([0, 1])
        ax1.set_xticks([0, 1])
        for x in range(2):
            for y in range(2):
                ax1.annotate(
                    str(round(100 * conf_matrix[x,y]/np.sum(conf_matrix[x]), 1)),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor='black',
                    color='white'
                )
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        ax1 = axes[1, 1]
        ax1.set_title("Test Conf.Mat")
        ax1.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        ax1.set_xticklabels(['Manual', 'Automatic'])
        ax1.set_yticklabels(['Manual', 'Automatic'])
        ax1.set_yticks([0, 1])
        ax1.set_xticks([0, 1])
        for x in range(2):
            for y in range(2):
                ax1.annotate(
                    str(round(100 * conf_matrix[x,y]/np.sum(conf_matrix[x]), 1)),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor='black',
                    color='white'
                )
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        plt.show()