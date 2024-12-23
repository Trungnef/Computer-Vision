import cv2
import numpy as np
from skimage.filters import gabor, gaussian, laplace

class ImageFiltering:
    def __init__(self, image):
        self.image = image

    def gaussian_blur(self, sigma):

        return cv2.GaussianBlur(self.image, (15, 15), sigma)

    def median_blur(self, kernel_size):
        #return cv2.medianBlur(self.image.astype(np.uint8), kernel_size)
        return cv2.medianBlur(self.image, kernel_size)
    def average_blur(self, kernel_size):
        return cv2.blur(self.image.astype(np.uint8), (kernel_size, kernel_size))


    def bilateral_filter(self):
        return cv2.bilateralFilter(self.image, 15, 75, 75)
    

    def gabor_filter(self, theta, frequency):
        if self.image.ndim == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image.astype(np.uint8)

        filtered, _ = gabor(gray_image, frequency=frequency, theta=theta)
        return (filtered * 255).astype(np.uint8)



    def laplacian_filter(self):
        if self.image.ndim == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image.astype(np.uint8)

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return np.abs(laplacian).astype(np.uint8)

    def sobel_filter(self):
        if self.image.ndim == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
        else:
             gray_image = self.image.astype(np.uint8)  # Chuyển sang ảnh xám

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return np.abs(sobel_combined).astype(np.uint8)
    

    def canny_filter(self, threshold1, threshold2):
        edges = cv2.Canny(self.image, threshold1, threshold2)
        return edges

    