import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def channel_without_fading(z, snr_db):
  z = z.to(device)

  # AWGN
  mean_TP=[]
  mean_NP=[]
  for i in range(z.shape[0]):
    mean_TP.append(torch.mean(z[i,:,:,:] ** 2))#送信平均電力
    mean_NP.append(mean_TP[i] / 10 ** (snr_db / 10)) #雑音平均電力

  
  noise = torch.empty(z.shape)
  for j in range(z.shape[0]):
    noise[j,:,:,:] = torch.normal(mean=0, std=(mean_NP[j].item())**0.5, size=(z.shape[1], z.shape[2], z.shape[3])) #雑音信号

  noise = noise.to(device)
  z_hat = z + noise

  return z_hat



def channel_with_fading(z, snr_db):
  z = z.to(device)
  
  # Rayleigh fading
  rf = torch.normal(mean=0, std=(1/2)**0.5, size=(z.shape)).to(device)#レイリーフェージング信号
  
  rf_z = z + rf

  # AWGN
  mean_TP=[]
  mean_NP=[]
  for i in range(z.shape[0]):
    mean_TP.append(torch.mean(rf_z[i,:,:,:] ** 2))#送信平均電力
    mean_NP.append(mean_TP[i] / 10 ** (snr_db / 10)) #雑音平均電力
  
  noise = torch.empty(z.shape)
  for j in range(z.shape[0]):
    noise[j,:,:,:] = torch.normal(mean=0, std=(mean_NP[j].item())**0.5, size=(z.shape[1], z.shape[2], z.shape[3])) #雑音信号

  noise = noise.to(device)
  z_D = rf_z + noise

  return z_D