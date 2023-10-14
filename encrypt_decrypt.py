import cv2
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
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#image with single pixel value different
normalized_image1 = normalized_image
normalized_image1[10][15] = normalized_image1[10][15]+5

# Convert to a NumPy arrayv
pixel_values = np.array(normalized_image, dtype=np.uint8).astype('int32')

def encrypt(image,x,y):
    key = Henon_Map.export_mask(x,y,img_size)
    en_img = np.bitwise_xor(image,key)
    return en_img


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

def decrypt(image,x,y):
    key = Henon_Map.export_mask(x,y,img_size)
    de_img = np.bitwise_xor(image,key)
    return de_img

def display_image_from_array(pixel_values,title):
    # Create an image from the 2D array
    image = cv2.merge([pixel_values])

    img = np.uint8(image)

    # Show the image (optional)
    cv2.imshow(title, img)

encrypt_img = encrypt(normalized_image,0.001,0.2)
decrypt_img = decrypt(encrypt_img,0.001,0.2)
encrypt_img_snp = encrypt_snp(normalized_image,0.001,0.2,5)
encrypt_img_snp1 = encrypt_snp(normalized_image1,0.001,0.2,5)
decrypt_img_snp = decrypt_snp(encrypt_img_snp,0.001,0.2,5)

def display_images():
    cv2.imshow("Original Image",image)
    display_image_from_array(encrypt_img,"encrypted Image")
    display_image_from_array(decrypt_img,"decrypted Image")
    display_image_from_array(encrypt_img_snp,"encrypted SNP Image")
    display_image_from_array(encrypt_img_snp1,"encrypted SNP 1 Image")
    display_image_from_array(decrypt_img_snp,"decrypted SNP Image")

display_images()
# diff = pixel_diff.pixel_difference(pixel_values,decrypt_img_snp)
# pixel_diff.display_image_from_array(diff,'Difference Image')
#
psnr_value = psnr_paramter.calculate_psnr(encrypt_img_snp,encrypt_img_snp1)
print(f"PSNR: {psnr_value:.2f} dB")
#NPCR between two cipher image with one different pixel in original image
print(npcr_parameter.npcr(encrypt_img_snp,encrypt_img_snp1 ))


#UACI between two cipher image with one different pixel in original image
print(uaci_parameter.uaci(encrypt_img_snp,encrypt_img_snp1))
cv2.waitKey(0)
