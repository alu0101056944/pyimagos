'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Dilation filter.
'''

from typing import Union

import torch
import cv2 as cv
import numpy as np

from src.image_filters.image_filter import ImageFilter

class DilationFilter(ImageFilter):
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
      self.kernel_size = kernel_size
      self.iterations = iterations

    def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
      erode_kernel = np.ones(self.kernel_size, np.uint8)
      return cv.erode(image, erode_kernel, self.iterations)

    def get_name(self) -> str:
      return f"DilationFilter(ksize={self.kernel_size}, itr={self.iterations})"

    def get_params(self) -> dict:
      return {"kernel_size": self.kernel_size, "iterations": self.iterations}
