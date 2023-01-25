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


def psnr_mean(img_1, img_2, data_range=255):
    psnr_element = []
    for i in range(img_1.shape[0]):
      mse = np.mean((img_1[i,:,:,:].astype(float) - img_2[i,:,:,:].astype(float)) ** 2)
      psnr_element.append(10 * np.log10((data_range ** 2) / mse))
    return np.mean(psnr_element)



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


# Image preprocessing modules
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

# CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]))

# Data loader
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
  num_epochs = 100

  comp_ratio=1/6
  snr_db = 20
  model = Net_FNN().to(device)

  criterion = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters())

  PATH = 'weight/cifar_net_snr20commpression-half100.pth'
  model.load_state_dict(torch.load(PATH)) # モデルを呼び出す場合
  # torch.save(model.state_dict(), PATH)
  # Test the model
  model.eval()
  with torch.no_grad():
      PSNR_element = 0
      total = 0
      for images, _ in test_loader:

        # correct = images.to('cpu').detach().numpy().copy()
        # correct = correct * 255
        # correct = correct.astype(np.uint8)
        # img = correct.transpose(0,2,3,1)
        # plt.imsave("./img0.png",img[0])

        images = images.to(device)
        outputs = model(images, snr_db)
        image = images.to('cpu').detach().numpy().copy()
        output = outputs.to('cpu').detach().numpy().copy()
        correct = image * 255
        correct = correct.astype(np.uint8)
        img = correct.transpose(0,2,3,1)
        prediction = output * 255
        prediction = prediction.astype(np.uint8)
        re_img = prediction.transpose(0,2,3,1)


        # 入力画像と復元画像の表示
        plt.subplot(1,2,1)
        plt.imshow(img[0],cmap="Greys")
        plt.title("original")
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1,2,2)
        plt.imshow(re_img[0],cmap="Greys")
        plt.title("restored")
        # plt.xlabel("SNR="+ str(SNR) + ", PSNR=" + str(PSNR))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.imsave("./img.png",img[0])
        plt.imsave("./re_img.png",re_img[0])


        PSNR_element += psnr_mean(correct, prediction)
        total += 1
          # PSNR
      # if (i + 1) % 100 == 0:
      print("SNR")
      print(snr_db)
      print('Accuracy of the model on the test images (PSNR): {}'.format(PSNR_element / total))