import cv2
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
from Differential_parameter import NPCR, UACI, modified_img, encrypted_image
import openpyxl
import pandas as pd
# Define the file path for the Excel file
file_path = "Metrics val.xlsx"

# Create a DataFrame filled with 0.0 values
data = np.zeros((0, 5))

# Create a DataFrame from the data array
df = pd.DataFrame(data)

# Set column names
df.columns = ["PSNR","MSE","SSIM","ENTROPY","CORRELATION"]


def run_tests(img1, img2, filename):
    PSNR = cv2.PSNR(img1, img2)
    df.at[filename, 'PSNR'] = PSNR
    print("PSNR:", PSNR)

    MSE = np.square(np.subtract(img1, img2)).mean()
    df.at[filename, 'MSE'] = MSE
    print("MSE:", MSE)

    (SSIM, diff) = skimage.metrics.structural_similarity(img1, img2, full=True)
    df.at[filename, 'SSIM'] = SSIM
    print("SSIM:", SSIM)

    entropy1 = skimage.measure.shannon_entropy(img1)
    entropy2 = skimage.measure.shannon_entropy(img2)
    # print("Entropy1:", entropy1)
    df.at[filename, 'ENTROPY'] = entropy2
    print("Entropy2:", entropy2)

    # Correlation Coefficient
    CC = np.corrcoef(img1.flatten(), img2.flatten())
    df.at[filename, 'CORRELATION'] = CC[0][1]
    print("Correlation coefficient:", CC)

    # Histogram variance analysis for an image
    bins = np.arange(-0.5, 255 + 1, 1)
    hist_o = np.histogram(img2.flatten(), bins=bins)
    A = hist_o[0]
    bins = np.arange(-0.5, 255 + 1, 1)
    # hist_emb= np.histogram(img2.flatten(), bins=bins)
    # B=hist_emb[0]
    va = []
    c = 0
    for i in range(len(A)):
        a = A[i]
        c = 0
        for j in range(len(A)):
            b = A[j]
            c1 = (np.square(a - b)) / 2
            c = c + c1
        va.append(c)

    hv = np.sum(va) / (255 * 255)
    df.at[filename, 'HISTOGRAM_VARIANCE'] = hv

    print("Histogram variance:", hv)

    histogram(img1, img2)


def diff_parameter(encrypt_img1, encrypt_img2):
    # NCPR parameter (Differential Analysis)
    npcr_obj = NPCR()
    a = npcr_obj.npcr(encrypt_img1, encrypt_img2)
    df.at[filename, 'NPCR'] = a

    print("NPCR : " + str(a))

    # UACI paramter (Differential Analysis)
    uaci_obj = UACI()
    b = uaci_obj.uaci(encrypt_img1, encrypt_img2)
    df.at[filename, 'UACI'] = b
    print("UACI : " + str(b))


def histogram(image1, image2):
    histr1 = cv2.calcHist([image1], [0], None, [256], [0, 256]).flatten()
    histr2 = cv2.calcHist([image2], [0], None, [256], [0, 256]).flatten()
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


# folder_path1 = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images"
folder_path1 = r"Images\test_images"
# folder_path2 = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images_en"
folder_path2 = r"Images\test_images_en"

for filename in os.listdir(folder_path1):
    image_path1 = os.path.join(folder_path1, filename)
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image_path2 = os.path.join(folder_path2, filename)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    run_tests(image1, image2,filename)
    encrypt1, encrypt2 = encrypted_image(image1,modified_img(image1),b"Password")
    diff_parameter(encrypt1,encrypt2)
    print("\n------------------------------------------------------------------------------------------------------\n")


df.to_excel(file_path, index=True)