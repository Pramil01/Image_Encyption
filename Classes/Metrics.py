import cv2
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt

def run_tests(img1,img2):

    PSNR = cv2.PSNR(img1,img2)
    print("PSNR:", PSNR)

    MSE = np.square(np.subtract(img1,img2)).mean()
    print("MSE:", MSE)


    (SSIM, diff) = skimage.metrics.structural_similarity(img1, img2, full=True)
    print("SSIM:", SSIM)

    entropy1 = skimage.measure.shannon_entropy(img1)
    entropy2 = skimage.measure.shannon_entropy(img2)
    #print("Entropy1:", entropy1)
    print("Entropy2:", entropy2)

    #Correlation Coefficient
    CC = np.corrcoef(img1.flatten(), img2.flatten())
    print("Correlation coefficient:", CC)

    #Histogram variance analysis for an image
    bins = np.arange(-0.5, 255+1,1)
    hist_o = np.histogram(img2.flatten(), bins=bins)
    A=hist_o[0]    
    bins = np.arange(-0.5, 255+1,1)
    #hist_emb= np.histogram(img2.flatten(), bins=bins)
    #B=hist_emb[0]
    va=[]
    c=0
    for i in range(len(A)):
        a=A[i]
        c=0
        for j in range (len(A)):  
            b=A[j]
            c1=(np.square(a-b))/2
            c=c+c1
        va.append(c)

    hv=np.sum(va)/(255*255)

    print ("Histogram variance:", hv)


    histogram(img1,img2)

def histogram(image1,image2):

    histr1 = cv2.calcHist([image1],[0],None,[256],[0,256]).flatten()
    histr2 = cv2.calcHist([image2],[0],None,[256],[0,256]).flatten()
    bins = np.arange(256)

    plt.figure(1)
    plt.title('Original')
    plt.xlabel('Grayscale')
    plt.ylabel('Frequency')
    plt.bar(bins, histr1, width=1.0, color='blue') 
    plt.xlim([0, 256])

    plt.figure(2)
    plt.title('Encrypted')
    plt.xlabel('Grayscale')
    plt.ylabel('Frequency')
    plt.bar(bins, histr2, width=1.0, color='blue')
    plt.xlim([0, 256])

    plt.show()

folder_path1 = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images"
folder_path2 = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images_en"

for filename in os.listdir(folder_path1):
        image_path1 = os.path.join(folder_path1, filename)
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image_path2 = os.path.join(folder_path2, filename)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        run_tests(image1,image2)