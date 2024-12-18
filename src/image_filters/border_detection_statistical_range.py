'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI. Detect borders through
  statistical range within sliding window.
'''
from typing import Union

import cv2 as cv
import numpy as np
from torch import Tensor

from src.image_filters.image_filter import ImageFilter

class BorderDetectionStatisticalRange(ImageFilter):
    def __init__(self, padding: int = 5):
        self.padding = padding

    def process(self, image: Union[np.array, Tensor]) -> np.array:
      blockSize = (3, 3)
      paddedImage = np.pad(image, self.padding, mode='reflect')
      
      bordersDetected = np.zeros_like(image, dtype=np.float32)

      for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = paddedImage[y:y + blockSize[0], x:x + blockSize[1]]
            statisticalRange = window.max() - window.min()
            bordersDetected[y, x] = statisticalRange

      return cv.normalize(bordersDetected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    def get_name(self):
        return 'GaussianFilter'

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}
