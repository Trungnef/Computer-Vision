import cv2
import numpy as np
from skimage.transform import resize, rotate, AffineTransform, warp, ProjectiveTransform, pyramid_gaussian

class GeometricTransformations:
    def __init__(self, image):
        self.image = image

    def resize_image(self, width, height):
        return cv2.resize(self.image, (width, height))

    def rotate_image(self, angle):
        rows, cols = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(self.image, M, (cols, rows))

    def affine_transform(self, src_points, dst_points):
        M = cv2.getAffineTransform(src_points, dst_points)
        return cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))

    def perspective_transform(self, src_points, dst_points):
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(self.image, M, (self.image.shape[1], self.image.shape[0]))

    def pyramid_transform(self):
        return cv2.pyrDown(self.image)


    def translate_image(self, tx, ty):
        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(self.image, M, (cols, rows))

    def crop_image(self, x1, y1, x2, y2):
        return self.image[y1:y2, x1:x2]

    def flip_image(self, flip_axis):
        return cv2.flip(self.image, flip_axis)