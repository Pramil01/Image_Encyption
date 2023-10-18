import numpy as np
import snp_de

class Decryption:

    def __init__(self) :
        pass
    
    # A Henon map of size n
    def Henon_Map(self,x,y,n,a=1.4,b=0.3):
        x_list = np.zeros(n)
        y_list = np.zeros(n)
        for i in range(n):
            x = 1 - a*x*x + y
            y = b*x
            x_list[i] = int((x + 1) * 127.5) % 256
            y_list[i] = int((y + 1) * 127.5) % 256
        return x_list,y_list

    # creating a mask of n * n by xor of x_list and y_list elements
    def create_mask(self,x_list,y_list,n):
        mask  = np.zeros((n,n),dtype=int)
        for i in range(n):
            for j in range(n):
                mask[i][j] = int(x_list[i]) ^ int(y_list[j])
        return mask 

    def get_mask(self,x,y,img_len):
        x_list,y_list = self.Henon_Map(x,y,img_len)
        mask =  self.create_mask(x_list,y_list,img_len)
        return mask
    
    def decrypt(self,image,x,y,N):
        img_size = len(image)
        key = self.get_mask(x,y,img_size)
        de_img = np.bitwise_xor(image,key)
        snp_de.snp_decryption(de_img,N)
        return de_img