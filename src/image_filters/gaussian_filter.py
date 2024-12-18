'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI
'''

from src.image_filters.image_filter import ImageFilter

import cv2 as cv
import numpy as np

class GaussianFilter(ImageFilter):
    def __init__(self, kernel_size: int = 5, sigma: float = 1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def process(self, image: np.array) -> np.array:
      return cv.GaussianBlur(image, (self.kernel_size, self.kernel_size),
                             self.sigma)

    def get_name(self):
        return 'GaussianFilter'

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}
