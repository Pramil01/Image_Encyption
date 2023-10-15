import cv2
import numpy as np


def pixel_difference(original_img,decrypted_img):
   
    # Check if the images have the same dimensions
    if original_img.shape != decrypted_img.shape:
        raise ValueError("Images must have the same dimensions")

    # Compute the absolute pixel-wise difference
    difference = cv2.absdiff(original_img, decrypted_img)
    return difference

def display_image_from_array(diff_img,title):
    print(np.ndarray.sum(diff_img))
    # Create an image from the 2D array
    image = cv2.merge([diff_img])
    img = np.uint8(image)

    # Show the image (optional)
    cv2.imshow(title, img)
