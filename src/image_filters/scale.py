'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Filter meant as a bandaid for lack of zoom options in the experiment GUI.
'''

from typing import Union

import torch
import cv2 as cv
import numpy as np

from src.image_filters.image_filter import ImageFilter

class ScaleFilter(ImageFilter):
    def __init__(self, scale_factor_x: float = 1, scale_factor_y: float = 1):
      self.scale_factor_x = scale_factor_x
      self.scale_factor_y = scale_factor_y

    def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
      return cv.resize(image, (0, 0), fx=self.scale_factor_x,
                       fy=self.scale_factor_y, interpolation=cv.INTER_LINEAR)

    def get_name(self) -> str:
      return f"Scale(factorX={self.scale_factor_x}, factorY={self.scale_factor_y})"

    def get_params(self) -> dict:
      return {"scale_factor_x": self.scale_factor_x,
              "scale_factor_y": self.scale_factor_y}
