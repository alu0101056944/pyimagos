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

class CannyFilter(ImageFilter):
  def __init__(self, threshold_1: int = 30, threshold_2: int = 135):
    self.threshold_1 = threshold_1
    self.threshold_2 = threshold_2

  def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    return cv.Canny(image, self.threshold_1, self.threshold_2)

  def get_name(self) -> str:
    return f"Canny(thresh1={self.threshold_1},thresh1={self.threshold_2})"

  def get_params(self) -> dict:
    return {"threshold_1": self.threshold_1, "threshold_2": self.threshold_2}
