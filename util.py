from calendar import c
import torch
import math
# import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zigzag(input):
  output = torch.empty([input.shape[0], input.shape[1], input.shape[2] * input.shape[3]])
  # output = torch.empty([input.shape[0] * input.shape[1], 3])
  for a in range(input.shape[0]):
    for b in range(input.shape[1]):
    
        #initializing the variables
        #----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input[a,b,:,:].shape[0]
        hmax = input[a,b,:,:].shape[1]
        
        #print(vmax ,hmax )

        i = 0

        #----------------------------------

        while ((v < vmax) and (h < hmax)):
          
            if ((h + v) % 2) == 0:                 # going up
                
                if (v == vmin):
                  #print(1)
                    output[a,b,i] = input[a,b,v,h]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                  #print(2)
                  output[a,b,i] = input[a,b,v,h] 
                  v = v + 1
                  i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                  #print(3)
                  output[a,b,i] = input[a,b,v,h] 
                  v = v - 1
                  h = h + 1
                  i = i + 1

            
            else:                                    # going down

              if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[a,b,i] = input[a,b,v,h] 
                h = h + 1
                i = i + 1
            
              elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[a,b,i] = input[a,b,v,h] 

                if (v == vmax -1):
                  h = h + 1
                else:
                  v = v + 1

                i = i + 1

              elif ((v < vmax -1) and (h > hmin)):     # all other cases
                #print(6)
                output[a,b,i] = input[a,b,v,h] 
                v = v + 1
                h = h - 1
                i = i + 1




            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
              #print(7)        	
              output[a,b,i] = input[a,b,v,h] 
              break

    #print ('v:',v,', h:',h,', i:',i)
  return output.to(device), vmax, hmax

def inverse_zigzag(input, vmax, hmax):
	
	#print input.shape
  output = torch.empty([input.shape[0], input.shape[1],vmax, hmax])

  for a in range(input.shape[0]):
    for b in range(input.shape[1]):
      # initializing the variables
      #----------------------------------
      h = 0
      v = 0

      vmin = 0
      hmin = 0


      i = 0
        #----------------------------------

      while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)   	
        if ((h + v) % 2) == 0:                 # going up
                
          if (v == vmin):
            #print(1)
            
            output[a, b, v, h] = input[a,b,i]        # if we got to the first line

            if (h == hmax):
              v = v + 1
            else:
              h = h + 1                        

            i = i + 1

          elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
            #print(2)
            output[a, b, v, h] = input[a,b,i] 
            v = v + 1
            i = i + 1

          elif ((v > vmin) and (h < hmax -1 )):    # all other cases
            #print(3)
            output[a, b, v, h] = input[a,b,i] 
            v = v - 1
            h = h + 1
            i = i + 1

            
        else:                                    # going down

          if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
            #print(4)
            output[a, b, v, h] = input[a,b,i] 
            h = h + 1
            i = i + 1
            
          elif (h == hmin):                  # if we got to the first column
            #print(5)
            output[a, b, v, h] = input[a,b,i] 
            if (v == vmax -1):
              h = h + 1
            else:
              v = v + 1
            i = i + 1
                            
          elif((v < vmax -1) and (h > hmin)):     # all other cases
            output[a, b, v, h] = input[a,b,i] 
            v = v + 1
            h = h - 1
            i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
          #print(7)        	
          output[a, b, v, h] = input[a,b,i] 
          break

  return output.to(device)


def reshape(block):

  out = torch.reshape(block, (block.shape[0], block.shape[1], 64, int(block.shape[2] / 64)))

  return out

def inverse_reshape(block):

  out = torch.reshape(block, (block.shape[0], block.shape[1], block.shape[2]*block.shape[3]))

  return out

def Hadamard_Transformation(block):
  n=8
  
  # アダマール行列の生成
  ha =  torch.Tensor([[1,1],[1,-1]])

  def sylvester(n):
      b = torch.Tensor(([1,1],[1,-1]))
      for i in range(n-1):
          b = torch.kron(ha,b)
      return b
  hadamard = sylvester(int(math.log2(n**2)))/n

  hadamard = hadamard.to(device)
  block = block.to(device)

  HT = torch.empty(block.shape).to(device)
  for i in range(block.shape[0]):
    for j in range(block.shape[1]):
        HT[i,j,:,:] = torch.mm(hadamard,block[i,j,:,:])
  return HT




def normalization(block, P):
  #ベクトル化
  z_tilta = torch.reshape(block, (block.shape[0], -1))
  # z_restore = torch.reshape(z,(block.shape))
  # qed = block - z_restore

  # 正規化
  k = int(z_tilta.shape[1] / 2)
  z = torch.empty(z_tilta.shape)
  
  for i in range(z.shape[0]):
    # a = z_tilta[i,:]
    z_p = torch.square(z_tilta[i,:])
    z_p = torch.sum(z_p)
    z_p = torch.sqrt(z_p)
    z[i,:]= (k * P) ** 0.5 * (z_tilta[i,:] / z_p)
  
  # mean = [] #送信電圧の平均
  # z_power = torch.empty([z.shape[0], int(z.shape[1] / 2)])
  # for a in range(z.shape[0]):
  #   for b in range(int(z.shape[1] / 2)):
  #     z_power[a,b]=torch.sqrt(z[a,2*b]**2 +z[a,2*b+1]**2)
  #   mean.append(torch.mean(z_power[a,:]))
  # zをblockと同じに形状に
  z = torch.reshape(z,(block.shape))

  return z


def fix_pixel(block):
  for l in range(block.shape[3]):
    for k in range(block.shape[2]):
          for j in range(block.shape[1]):
              for i in range(block.shape[0]):
                  if block[i,j,k,l] > 1.0:
                      block[i,j,k,l] = 1.0
                      # print("a")
                  elif block[i,j,k,l] < 0.0:
                      block[i,j,k,l] = 0.0
                      # print("b")
                  else:
                      block[i,j,k,l] = block[i,j,k,l]

    return block

def change_normal(dct):
   # 正規化
  dct = dct.cpu()
  dct_min = torch.empty(dct.shape[0],dct.shape[1]) #最小値の取得
  for i in range(dct.shape[0]):
    for j in range(dct.shape[1]):
      dct_min[i,j] = torch.min(dct[i,j]) 

  dct_max = torch.empty(dct.shape[0],dct.shape[1]) #最大値の取得
  for i in range(dct.shape[0]):
    for j in range(dct.shape[1]):
      dct_max[i,j]=torch.max(dct[i,j])
  
  for i in range(dct.shape[0]):
    for j in range(dct.shape[1]):
        for k in range(dct.shape[2]):
              dct[i,j,k] = (dct[i,j,k] - dct_min[i,j]) / (dct_max[i,j] - dct_min[i,j])
              
  return dct.to(device)