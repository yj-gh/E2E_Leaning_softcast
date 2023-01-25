import torch
import torch.nn as nn
import torch.nn.functional as F
from e2e_channel import channel_without_fading, channel_with_fading
# from util import DCT2, IDCT2
from _dct import dct, idct, dct1, idct1, dct_2d, idct_2d
from util import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net_FNN(nn.Module):
  #利用するネットワークの種類
  def __init__(self, c):
    super().__init__()
    
    self.liner1 = nn.Linear(1024, 1000) #FNN：（入力データ数，出力データ数）
    self.liner2 = nn.Linear(1000, 1000) #FNN：（入力データ数，出力データ数）
    self.liner3 = nn.Linear(1000, 1000) #FNN：（入力データ数，出力データ数）
    self.liner4 = nn.Linear(1000, int(1024*c)) #FNN：（入力データ数，出力データ数）
    self.liner5 = nn.Linear(int(1024*c), 1000) #FNN：（入力データ数，出力データ数）
    self.liner6 = nn.Linear(1000, 1024) #FNN：（入力データ数，出力データ数）
    
    # self.act = nn.ReLU()
    self.act = nn.PReLU()
    # self.act = nn.Tanh()
    

  # ネットワーク構図の定義
  def forward(self, img, snr_db, c):

    #疑似アナログ(エンコーダーとデコーダー全体を学習)
    #エンコーダー
    # img = img * 255

    # DCT変換
    dct2= dct_2d(img)

    # dct_list, vmax, hmax = zigzag(dct2) #ジグザグスキャン
    dct_list = torch.reshape(dct2,(dct2.shape[0], dct2.shape[1], dct2.shape[2] * dct2.shape[3])) #ベクトル化

    # dct_list = change_normal(dct_list)

    # x = torch.empty(dct_list.shape)
    x = torch.empty(dct2.shape[0], dct2.shape[1], int((dct2.shape[2] * dct2.shape[3])*c))

    # RGBそれぞれをFNNでスケーリング
    x_R = self.act(self.liner1(dct_list[:,0]))
    x_R = self.act(self.liner2(x_R))
    x_R = self.act(self.liner3(x_R))
    x[:,0] = self.liner4(x_R)

    x_G = self.act(self.liner1(dct_list[:,1]))
    x_G = self.act(self.liner2(x_G))
    x_G = self.act(self.liner3(x_G))
    x[:,1] = self.liner4(x_G)

    x_B = self.act(self.liner1(dct_list[:,2]))
    x_B = self.act(self.liner2(x_B))
    x_B = self.act(self.liner3(x_B))
    x[:,2] = self.liner4(x_B)

    # ベクトルをテンソル化
    X = reshape(dct_list)

    # アダマール変換
    y = Hadamard_Transformation(X)

    # 電力割り当てのための正規化
    z = normalization(y, P =1)

    
    #チャネル（通信路）
    z_hat = channel_without_fading(z, snr_db)


    #デコーダー
    # アダマール変換
    re_X = Hadamard_Transformation(y)

    #テンソルをベクトル化
    re_dct_list = inverse_reshape(re_X)

    x_hat = torch.empty(dct_list.shape)

    # RGBそれぞれをFNNでリスケーリング
    x_R_hat = self.act(self.liner5(re_dct_list[:,0]))
    x_R_hat = self.act(self.liner2(x_R_hat))
    x_R_hat = self.act(self.liner3(x_R_hat))
    x_hat[:,0] = self.liner6(x_R_hat)

    x_G_hat = self.act(self.liner5(re_dct_list[:,1]))
    x_G_hat = self.act(self.liner2(x_G_hat))
    x_G_hat = self.act(self.liner3(x_G_hat))
    x_hat[:,1] = self.liner6(x_G_hat)

    x_B_hat = self.act(self.liner5(re_dct_list[:,2]))
    x_B_hat = self.act(self.liner2(x_B_hat))
    x_B_hat = self.act(self.liner3(x_B_hat))
    x_hat[:,2] = self.liner6(x_B_hat)
    
    #ベクトルを入力と同じテンソルの形に変換
    # re_dct = inverse_zigzag(x_hat, vmax, hmax)
    re_dct = torch.reshape(re_dct_list, (dct2.shape))
    # re_dct = re_dct * 32.0

    #IDCT変換して画素値に
    out = idct_2d(re_dct)

    # out = re_img / 255

    return out.to(device)
