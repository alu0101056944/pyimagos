'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI. Blurs the image.
  kernel size and sigma impact blur intensity.
'''
from typing import Union

import cv2 as cv
import numpy as np
from torch import Tensor

from src.image_filters.image_filter import ImageFilter

class GaussianFilter(ImageFilter):
    def __init__(self, kernel_size: int = 5, sigma: float = 1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def process(self, image: Union[np.array, Tensor]) -> np.array:
      return cv.GaussianBlur(image, (self.kernel_size, self.kernel_size),
                             self.sigma)

    def get_name(self):
        return 'GaussianFilter'

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}
