'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI
'''

from typing import Union

from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor

class ImageFilter(ABC):
  @abstractmethod
  def process(self, image: Union[np.array, Tensor]) -> np.array:
    pass

  @abstractmethod
  def get_name(self) -> str:
    pass

  abstractmethod
  def get_params(self) -> dict:
    pass
