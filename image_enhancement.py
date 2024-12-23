import cv2
import numpy as np
from skimage import exposure
from sklearn.cluster import KMeans

class ImageEnhancement:
    def __init__(self, image):
        self.image = image

    def adjust_brightness_contrast(self, brightness, contrast):
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=1 + contrast / 100, beta=brightness)
        return adjusted_image

    def gamma_correction(self, gamma):
        gamma_corrected = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        return gamma_corrected

    def histogram_equalization(self):
        if self.image.ndim == 3:  # Ảnh màu
            equalized_image = np.zeros_like(self.image)
            for channel in range(3):
                equalized_image[:, :, channel] = cv2.equalizeHist(self.image[:, :, channel].astype(np.uint8))
        else: # Ảnh xám
            equalized_image = cv2.equalizeHist(self.image.astype(np.uint8))
        return equalized_image



    def color_quantization(self, n_colors):
        pixels = self.image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
        quantized_image = kmeans.cluster_centers_[kmeans.labels_]
        quantized_image = quantized_image.reshape(self.image.shape).astype(np.uint8)
        return quantized_image