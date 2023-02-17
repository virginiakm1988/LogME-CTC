import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from torchvision import datasets 
import pandas as pd
from torch.utils.data import DataLoader, Dataset 
import time

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([                          
    transforms.ToTensor(),
])

batch_size = 32
dataset = datasets.CIFAR10('../data', train=True, download=True,
                       transform=train_transform)
test_set = datasets.CIFAR10('../data', train=False,
                       transform=test_transform)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8,0.2])
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, num_workers=5, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size, num_workers=5)

from tqdm import tqdm
model = models.resnet18(pretra).cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001) # optimizer 使用Adagrad
num_epoch = 20

#adam (momemtum)

best_acc = 0

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
        if val_acc/val_set.__len__() > best_acc:
            torch.save(model.state_dict, "./models/model_cifar10.best.ckpt")