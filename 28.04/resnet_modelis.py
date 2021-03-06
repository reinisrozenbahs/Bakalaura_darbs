import os
import pickle
import random

import torch.nn.functional as F

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
import torchvision.transforms as transforms
from torch.hub import download_url_to_file
from tqdm import tqdm

import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

batch_size_train = 64
batch_size_test = 1000

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

data_loader_train = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

data_loader_test = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../../../data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(stride, stride), padding=(1,1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels,
                                         kernel_size = (1, 1),
                                         stride = (stride, stride), bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1.forward(x)
        out = F.relu(out)
        out = self.bn1.forward(out)
        out = self.conv2.forward(out)

        if self.is_bottleneck:
            residual = self.shortcut.forward(x)

        out += residual
        out = F.relu(out)
        out = self.bn2.forward(out)

        return out

class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.bn.forward(F.relu(self.conv.forward(x)))
        out = self.avg_pool(out)
        return out



class ModelResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4,
                                     kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=4)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.identity_block_1 = ResBlock(in_channels=4, out_channels=4)
        self.identity_block_2 = ResBlock(in_channels=4, out_channels=4)

        self.bottleneck_block_1 = ResBlock(in_channels=4, out_channels=8, stride=2)
        self.identity_block_3 = ResBlock(in_channels=8, out_channels=8)

        self.bottleneck_block_2 = ResBlock(in_channels=8, out_channels=16, stride=2)
        self.identity_block_4 = ResBlock(in_channels=16, out_channels=16)

        self.bottleneck_block_3 = ResBlock(in_channels=16, out_channels=32, stride=2)
        self.identity_block_5 = ResBlock(in_channels=32, out_channels=32)

        self.linear = torch.nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.max_pool.forward(out)

        out = self.identity_block_1.forward(out)
        out = self.identity_block_2.forward(out)

        out = self.bottleneck_block_1.forward(out)
        out = self.identity_block_3.forward(out)

        out = self.bottleneck_block_2.forward(out)
        out = self.identity_block_4.forward(out)

        out = self.bottleneck_block_3.forward(out)
        out = self.identity_block_5.forward(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)
        return out

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=5)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1))
        out = out.view(x.size(0), -1)
        out = F.softmax(out, dim=1)
        return out


model = ModelResnet()
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

def createFile(path,filename):
    name = path+'/'+filename  # Name of text file coerced with +.txt
    try:
        file = open(name,'a')   # Trying to create a new file or open one
        file.close()
    except:
        print('File not created')

def writeToFile(path, value):
    try:
        file = open(path,'a')
        addToFile = value
        file.write("\n" + addToFile)
        file.close()
    except:
        print('data not written in file')

filepath_loss = "/Users/reinisrozenbahs/BD_dati_txt"
createFile(filepath_loss, "demo.txt")

for epoch in range(1, 5):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):
            # #CAUTION random resize here!!! model must work regardless
            # out_size = int(28 * (random.random() * 0.3 + 1.0))
            # x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(out_size, out_size))

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
            print(f'accuracy: {str(acc)}')
            writeToFile("/Users/reinisrozenbahs/BD_dati_txt/demo.txt", str(acc))

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')
        random_string = ' '.join(metrics_strs)
        #writeToFile("/Users/reinisrozenbahs/BD_dati_txt/demo.txt", random_string)

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

    conf_matrix = np.zeros((idx_y_prim.size, idx_y.size))
    for idx in range(idx_y_prim.size):
        i = idx_y_prim[idx]
        j = idx_y[idx]
        conf_matrix[i,j] +=1

    print(conf_matrix)

    for i in range(idx_y_prim.size):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        TP += conf_matrix[i, i]
        FP += conf_matrix[i,:].sum() - conf_matrix[i,i]
        FN += conf_matrix[:,i].sum() - conf_matrix[i,i]
        TN += conf_matrix[:,:].sum() - conf_matrix[i,:].sum() - conf_matrix[:,i].sum()
        F_Score = TP / (TP + 0.5 * (FP * FN))
        #print(TP, FP, FN, TN)
        #print(F_Score)

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.1)

input('quit?')
