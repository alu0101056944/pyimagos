'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Gaussian filter that blurs the image.
'''

from typing import Union

import torch
import cv2 as cv
import numpy as np

from src.image_filters.image_filter import ImageFilter

class GaussianBlurFilter(ImageFilter):
    def __init__(self, kernel_size: int = 10, sigma: float = 1):
      self.kernel_size = kernel_size
      self.sigma = sigma

    def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
      return cv.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)

    def get_name(self) -> str:
      return f"GaussianBlur(ksize={self.kernel_size},sig={self.sigma})"

    def get_params(self) -> dict:
      return {"kernel_size": self.kernel_size, "sigma": self.sigma}
