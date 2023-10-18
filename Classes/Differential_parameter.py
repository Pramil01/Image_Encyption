import numpy as np
import random
from Encryption import Encryption

def encrypted_image(img1,img2):
    en_obj = Encryption()
    en_img1 = en_obj.encrypt(img1, 0.001, 0.3, 2)
    en_img2 = en_obj.encrypt(img2, 0.001, 0.3, 2)
    return en_img1,en_img2
def modified_img(img):
    i = random.randint(0,len(img))
    j = random.randint(0,len(img[0]))
    img[i][j] += random.randint(0,255)
    return img

# NPCR(Number of Pixel Change Rate)
class NPCR:
    def __init__(self):
        pass

    def sumofpixelval(self, height, width, img1, img2):
        matrix = np.empty([width, height])

        for y in range(0, height):
            for x in range(0, width):
                if img1[x, y] == img2[x, y]:
                    matrix[x, y] = 0  # pixel values are same
                else:
                    matrix[x, y] = 1  # pixel values are different
        psum = 0

        for y in range(0, height):
            for x in range(0, width):
                psum = matrix[x, y] + psum  # sum all matrix values
        return psum

    def npcr(self, img1, img2):

        height = img1.shape[0]
        width = img2.shape[1]
        npcrv = ((self.sumofpixelval(height, width, img1, img2) / (height * width)) * 100)
        return npcrv


# UACI(Unified Average Changing Intensity)
class UACI:
    def __init__(self):
        pass

    def uaci(self, img1, img2):
        height, width = img1.shape
        value = 0
        for y in range(height):
            for x in range(width):
                value += (abs(int(img1[x, y]) - int(img2[x, y])))
        value = value * 100 / (width * height * 255)
        return value
