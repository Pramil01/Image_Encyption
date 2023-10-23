import numpy as np
import snp_en
import hashlib
import struct


def generate_henon_parameters(secret_key):
    hash_obj = hashlib.sha256(secret_key)
    hash_bytes = hash_obj.digest()[:16]

    # Split the hash into four 32-bit integers (x, y, a, b)
    x, y, a, b = struct.unpack('>IIII', hash_bytes)

    # Map x and y to the desired range (e.g., between -1 and 1)
    x = -1 + (x / 0xFFFFFFFF) * 2
    y = -1 + (y / 0xFFFFFFFF) * 2

    # Map a and b to the desired range (e.g., between 1 and 2)
    a = 1 + (a / 0xFFFFFFFF)
    b = 1 + (b / 0xFFFFFFFF)
    return x,y,a,b


class Encryption:

    # m= hashlib.sha256()
    def __init__(self) :
        pass
    
    # A Henon map of size n
    def Henon_Map(self,x,y,n,a,b):
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

    def get_mask(self,x,y,img_len,a,b):
        x_list,y_list = self.Henon_Map(x,y,img_len,a,b)
        mask =  self.create_mask(x_list,y_list,img_len)
        return mask
    
    def encrypt(self,image,x,y,N,a,b):
        img_size = len(image)
        key = self.get_mask(x,y,img_size,a,b)
        snp_en.snp_encryption(image,N)
        en_img = np.bitwise_xor(image,key)
        return en_img

