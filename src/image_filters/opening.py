'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Opening filter.
'''

from typing import Union

import torch
import cv2 as cv
import numpy as np

from src.image_filters.image_filter import ImageFilter

class OpeningFilter(ImageFilter):
    def __init__(self, kernel_size: int = 3):
      self.kernel_size = kernel_size

    def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
      opening_kernel = np.ones(self.kernel_size, np.uint8)
      return cv.morphologyEx(image, cv.MORPH_OPEN, opening_kernel)

    def get_name(self) -> str:
      return f"OpeningFilter(ksize={self.kernel_size})"

    def get_params(self) -> dict:
      return {"kernel_size": self.kernel_size, "iterations": self.iterations}
