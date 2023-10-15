import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import Henon_Map
import snp_en
import snp_de
import pixel_diff
import psnr_paramter
import npcr_parameter
import uaci_parameter
 
# Load the grayscale image
image = cv2.imread('Lena_128.jpg', cv2.IMREAD_GRAYSCALE)
img_size = len(image)
# Normalize pixel values to the range 0-255
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('int32')
#image with single pixel value different
normalized_image1 = normalized_image
normalized_image1[10][15] = normalized_image1[10][15]+5

# Convert to a NumPy arrayv
# pixel_values = np.array(normalized_image, dtype=np.uint8)


def encrypt_snp(image,x,y,N):
    key = Henon_Map.export_mask(x,y,img_size)
    snp_en.snp_encryption(image,N)
    en_img = np.bitwise_xor(image,key)
    return en_img

def decrypt_snp(image,x,y,N):
    key = Henon_Map.export_mask(x,y,img_size)
    de_img = np.bitwise_xor(image,key)
    snp_de.snp_decryption(de_img,N)
    return de_img



def display_image_from_array(pixel_values,title):
    # Create an image from the 2D array
    image = cv2.merge([pixel_values])

    img = np.uint8(image)

    # Show the image (optional)
    cv2.imshow(title, img)

now_start = datetime.now()
encrypt_img_snp = encrypt_snp(normalized_image,0.001,0.2,5)
now_end = datetime.now()
print(now_end - now_start)

encrypt_img_snp1 = encrypt_snp(normalized_image1,0.001,0.2,5)
decrypt_img_snp = decrypt_snp(encrypt_img_snp,0.001,0.2,5)

def display_images():
    cv2.imshow("Original Image",image)

    display_image_from_array(encrypt_img_snp,"encrypted SNP Image")
    display_image_from_array(encrypt_img_snp1,"encrypted SNP 1 Image")
    display_image_from_array(decrypt_img_snp,"decrypted SNP Image")

def display_stats():
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('int32')

    display_images()
    diff = pixel_diff.pixel_difference(normalized_image,decrypt_img_snp)
    pixel_diff.display_image_from_array(diff,'Difference Image')

    psnr_value = psnr_paramter.calculate_psnr(normalized_image,decrypt_img_snp)
    print(f"PSNR: {psnr_value:.2f} dB")
    #NPCR between two cipher image with one different pixel in original image
    print(npcr_parameter.npcr(encrypt_img_snp,encrypt_img_snp1 ))

    #UACI between two cipher image with one different pixel in original image
    print(uaci_parameter.uaci(encrypt_img_snp,encrypt_img_snp1))

def add_noise(image,std):
    # Define the mean (loc) and standard deviation (scale) of the Gaussian noise
    mean = 0
    stddev = std  # You can adjust this value to control the noise level

    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(mean, stddev, image.shape).astype(np.int32)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    # Clip the values to be within the valid image pixel range (0-255)
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image

# noisy_image = add_noise(encrypt_img_snp,1)
# decrypt_noisy_image = decrypt_snp(noisy_image,0.001,0.2,5)
# display_image_from_array(noisy_image ,"Noisy Decrypted SNP Image")
# display_image_from_array(decrypt_noisy_image ,"Noisy Decrypted SNP Image")

def plot_noise_vs_pixel_diff(en_image,original_image):
    normalized_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype('int32')
    y = []
    std_array = []
    for i in range(11):
        std = i*0.1
        std_array.append(std)
        noisy_image = add_noise(en_image,std)
        decrypt_noisy_image = decrypt_snp(noisy_image,0.001,0.2,5)
        diff = npcr_parameter.npcr(normalized_image,decrypt_noisy_image)
        y.append(diff)
    plt.figure()
    plt.title('Gaussian Noise vs Pixel Difference between original and decypted image')
    plt.xlabel('Standard Deviation')
    plt.ylabel('pecentage of Number of different pixels(%)')
    plt.plot(std_array,y,"x") 
    plt.show()

plot_noise_vs_pixel_diff(encrypt_img_snp,image)

#display_stats()
cv2.waitKey(0)
