import os
import cv2
import numpy as np
from Encryption import Encryption
from Decryption import Decryption

def load_images(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        en_obj = Encryption()
        de_obj = Decryption()

        en_img = en_obj.encrypt(image,0.001,0.3,2)
        de_img = de_obj.decrypt(en_img,0.001,0.3,2)

        directory_en = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images_en"
        directory_de = r"F:\Miscellaneous\Minor Project\Classes\Images\test_images_de"

        os.chdir(directory_en)
        store_image_from_array(filename, en_img)

        os.chdir(directory_de)
        store_image_from_array(filename, de_img)
        

def store_image_from_array(title,pixel_values):
    # Create an image from the 2D array
    image = cv2.merge([pixel_values])

    img = np.uint8(image)

    # Show the image (optional)
    cv2.imwrite(title, img)

load_images(r"F:\Miscellaneous\Minor Project\Classes\Images\test_images")