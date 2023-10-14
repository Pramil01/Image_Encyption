import numpy as np


def sumofpixelval (height, width, img1, img2):
    matrix = np.empty([width, height])

    for y in range(0, height):
        for x in range(0, width):
            if img1[x,y] == img2[x,y]:
                matrix[x,y]=0 #pixel values are same

            else:
                matrix[x,y]=1 #pixel values are different

    psum=0

    for y in range(0, height):
        for x in range(0, width):
            psum = matrix[x, y] + psum  # sum all matrix values
    return psum

def npcr(img1, img2):

    height=img1.shape[0]
    width=img2.shape[1]
    npcrv=((sumofpixelval (height, width, img1, img2)/(height*width))*100)
    return npcrv

