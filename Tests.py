import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import encrypt_decrypt


image1 = cv2.imread('Lena_128.jpg', cv2.IMREAD_GRAYSCALE)
normalized_image = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX)
image2 = encrypt_decrypt.encrypt_snp(normalized_image,0.001,0.2,5)
image3 = encrypt_decrypt.decrypt_snp(image2,0.001,0.3,5)
image4 = encrypt_decrypt.decrypt_snp(image2,0.001,0.2,5)
def histogram(image1,image2):

    histr1 = cv2.calcHist([image1],[0],None,[256],[0,256]) 
    histr2 = cv2.calcHist([image2],[0],None,[256],[0,256]) 

    plt.figure(1)
    plt.title('With SNP')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histr1) 
    plt.xlim([0, 256])

    plt.figure(2)
    plt.title('Without SNP')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histr2) 
    plt.xlim([0, 256])

    plt.show()

def  Pearson_Correlation_Coefficient(image1,image2):
    data1 = image1.flatten()
    data2 = image2.flatten()

    correlation = np.corrcoef(data1, data2)[0, 1]

    return abs(correlation)

def cal_correlation_with_N(image,N):
    y = []
    for i in tqdm(range(N+1)):
        image_en = encrypt_decrypt.encrypt_snp(normalized_image,0.001,0.2,i)
        y.append(abs(Pearson_Correlation_Coefficient(image,image_en)))

    plt.figure()
    plt.title('Change in Correlation with N')
    plt.xlabel('Value of N')
    plt.ylabel('Correlation')
    plt.plot(range(N+1),y)

    plt.show()

# 0cal_correlation_with_N(image1,25)
print("Original and Encrypted Correlation: "+str(Pearson_Correlation_Coefficient(image1,image2)))
print("Original and Decrypted Correlation: "+str(Pearson_Correlation_Coefficient(image1,image4)))
print("Original and Decrypted with slightly diff. parameter Correlation: "+str(Pearson_Correlation_Coefficient(image1,image3)))
