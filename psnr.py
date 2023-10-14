import cv2
import numpy as np

def calculate_psnr(image1, image2):
    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # If MSE is close to zero, return a very high PSNR value (infinite)
    if mse == 0:
        return float('inf')

    # Calculate the maximum possible pixel value as we are dealing with Grayscale image
    max_pixel_value = 255.0

    # Calculate the PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr


