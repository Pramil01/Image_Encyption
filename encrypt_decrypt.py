import cv2
import numpy as np
import Henon_Map

# Load the grayscale image
image = cv2.imread('Lena_128.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original Image",image)

  
# Normalize pixel values to the range 0-255
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# Convert to a NumPy arrayv
pixel_values = np.array(normalized_image, dtype=np.uint8)


mask = Henon_Map.export_mask(0.001,0.2,128)

def encrypt(image,mask):
    en_img = np.bitwise_xor(image,mask)
    return en_img

def decrypt(image,x,y):
    mask = Henon_Map.export_mask(x,y,128)
    de_img = np.bitwise_xor(image,mask)
    return de_img

def display_image_from_array(pixel_values,title):
    # Create an image from the 2D array
    image = cv2.merge([pixel_values])

    img = np.uint8(image)

    # Show the image (optional)
    cv2.imshow(title, img)

encrypt_img = encrypt(normalized_image,mask)
decrypt_img = decrypt(encrypt_img,0.001,0.2)
display_image_from_array(encrypt_img,"encrypted Image")
display_image_from_array(decrypt_img,"decrypted Image")

cv2.waitKey(0)