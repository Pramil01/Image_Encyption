import os
import cv2
import numpy as np
import Encryption

from Decryption import Decryption



def load_images(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        x,y,a,b = Encryption.generate_henon_parameters(b"Password")


        en_obj = Encryption.Encryption()
        de_obj = Decryption()


        en_img = en_obj.encrypt(image,x,y,2,a,b)
        de_img = de_obj.decrypt(en_img,x,y,2,a,b)
        #
        directory_en = r"C:\Users\Hp\PycharmProjects\Image_Encyption2\Classes\Images\test_images_en"
        directory_de = r"C:\Users\Hp\PycharmProjects\Image_Encyption2\Classes\Images\test_images_de"


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

load_images(r"C:\Users\Hp\PycharmProjects\Image_Encyption2\Classes\Images\test_images")