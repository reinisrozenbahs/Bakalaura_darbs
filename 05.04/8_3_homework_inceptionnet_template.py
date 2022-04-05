import os
import pickle
import random

import torch.nn.functional as F

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')


import torch.utils.data
import scipy.misc
import scipy.ndimage


USE_CUDA = torch.cuda.is_available()
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
MAX_LEN = 200 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
if USE_CUDA:
    MAX_LEN = None

class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../../../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = torch.from_numpy(np.array(X))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = F.one_hot(Y)

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y = self.Y[idx]

        return x, y


dataset_full = DatasetApples()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class InceptionBlockA(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
                                     kernel_size=(1, 1), bias=False)

        self.conv2_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
                                     kernel_size=(1, 1), bias=False)

        self.conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=(5, 5), padding=(2,2), bias=False)

        self.conv3_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
                                     kernel_size=(1, 1), bias=False)

        self.conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                       kernel_size=(3, 3), padding=(1,1), bias=False)

        self.conv3_3 = torch.nn.Conv2d(in_channels=64, out_channels=128,
                                       kernel_size=(3, 3), padding=(1,1), bias=False)

        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, padding=1)

        self.conv4 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
                                     kernel_size=(1, 1), bias=False)

        self.bn = torch.nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):

        conv1 = self.conv1.forward(x)
        conv2 = self.conv2_2.forward(self.conv2_1.forward(x))
        conv3 = self.conv3_3.forward(self.conv3_2.forward(self.conv3_1.forward(x)))
        conv4 = self.conv4.forward(self.avg_pool.forward(x))
        output = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        output = F.relu(output)
        output = self.bn.forward(output)

        return output


class InceptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.conv2d(in_channels=3, out_channels=32,
                                    kernel_size=(5, 5), bias=False)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.conv2d(in_channels=256, out_channels=32,
                                    kernel_size=(5, 5), bias=False)
        self.in_block_1 = InceptionBlockA()
        self.in_block_2 = InceptionBlockA()

        self.linear = torch.nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.max_pool.forward(out)
        out = F.relu(out)

        out = self.in_block_1.forward(out)
        out = self.conv2.forward(out)

        out = self.max_pool.forward(out)
        out = F.relu(out)

        out = self.in_block_2.forward(out)
        out = self.linear.forward(out)

        out = F.softmax(out)

        return out


model = InceptionNet()
loss_func = LossCrossEntropy()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

if USE_CUDA:
    model = model.to('cuda:0')
    # model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):

            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()

            y_prim = model.forward(x)
            loss = loss_func.forward(y, y_prim)
            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            np_y_prim = y_prim.data.numpy()
            np_y = y.data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
        plt.subplot(3, 6, j)
        color = 'green' if idx_y_prim[i]==idx_y[i] else 'red'
        plt.title(f"pred: {idx_y_prim[i]}\n real: {idx_y[i]}", c=color)
        plt.imshow(x[i].permute(1, 2, 0))

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.1)

input('quit?')
