import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from model import *
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


def print_info(info, max_epoch):
    print(f"epoch: {info['epoch']}/{max_epoch}", end=" ")
    print(f"[Loss]: {info['loss']:.2f}", end=" ")


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b' , '--batch_size', default=64, type=int)
parser.add_argument('-d','--depth', default=20, type=int,
                    metavar='L', help='The depth of the network')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Image preprocessing modules
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                             transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1000,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=0)


# # 訓練画像の表示                                         
# def imshow(img):
#     # img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.imsave("./a/test.png", np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(train_loader)
# images, _ = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))



if __name__ == '__main__':

    # set params
    args = parser.parse_args()
    lr = args.learning_rate
    num_epochs = 10000
    snr_db = 5
    comp_ratio=1

    model = Net_FNN(comp_ratio).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    total_step = len(train_loader)
    history = []
    for epoch in range(num_epochs):
        if epoch > 100:
            optimizer.param_groups[0]['lr'] = 0.0001
        model.train()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        for i, (images, _) in enumerate(train_loader):
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images, snr_db, comp_ratio)

            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 処理経過の観察
            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}\n"
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # 学習過程の記録
        info = {
            "epoch": epoch+1,
            "loss": float(loss),
        }
        history.append(info)
        print_info(info, num_epochs)

        # #nochannel
        # dir_weigth = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/nochannel/'
        # if not os.path.exists(dir_weigth):
        #     os.makedirs(dir_weigth)
        # PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/nochannel/epoch'+str(epoch+1)+'.pth'
        # torch.save(model.state_dict(), PATH)

        #AWGN
        dir_weigth = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/AWGN/snr'+str(snr_db)
        if not os.path.exists(dir_weigth):
            os.makedirs(dir_weigth)
        PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/AWGN/snr'+str(snr_db)+'/epoch'+str(epoch+1)+'.pth'
        torch.save(model.state_dict(), PATH)

        # #Fading
        # dir_weigth = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/Fading/snr'+str(snr_db)
        # if not os.path.exists(dir_weigth):
        #     os.makedirs(dir_weigth)
        # PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/Fading/snr'+str(snr_db)+'/epoch'+str(epoch+1)+'.pth'
        # torch.save(model.state_dict(), PATH)
    
    # #nochannel
    # dir_weigth = './history_loss/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/nochannel'
    # if not os.path.exists(dir_weigth):
    #     os.makedirs(dir_weigth)
    # history_path = './history_loss/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/nochannel/nosnr.csv'
    # # 学習経過を保存する。
    # pd.DataFrame(history).to_csv(history_path, index=False)

    #AWGN
    dir_weigth = './history_loss/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/AWGN/snr'+str(snr_db)
    if not os.path.exists(dir_weigth):
        os.makedirs(dir_weigth)
    history_path = './history_loss/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/AWGN/snr'+str(snr_db)+'.csv'
    # 学習経過を保存する。
    pd.DataFrame(history).to_csv(history_path, index=False)

    # #fading
    # history_path = './history_loss/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/Fading/snr'+str(snr_db)+'.csv'
    # # 学習経過を保存する。
    # pd.DataFrame(history).to_csv(history_path, index=False)


    # #nochannel
    # PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/nochannel/final.pth'
    # torch.save(model.state_dict(), PATH)

    #AWGN
    PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/AWGN/snr'+str(snr_db)+'/final.pth'
    torch.save(model.state_dict(), PATH)

    # #fading
    # PATH = './weight/cifar10/Compression_Ratio_'+str(int(comp_ratio*1000)/1000)+'/Fading/snr'+str(snr_db)+'/final.pth'
    # torch.save(model.state_dict(), PATH)