'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI
'''

from abc import ABC, abstractmethod

import numpy as np

class ImageFilter(ABC):
    @abstractmethod
    def process(self, image: np.array) -> np.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    abstractmethod
    def get_params(self) -> dict:
        pass
