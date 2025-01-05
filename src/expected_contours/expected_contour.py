'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours.

The AllowedLineSideBasedOnY enum decides which side of the two sides
that the line has is the space where contours relative to this
expected contour are allowed to be.
'''

from abc import ABC, abstractmethod
from typing import List
from enum import Enum, auto

class AllowedLineSideBasedOnY(Enum):
  GREATER = auto(),
  LOWER = auto(),
  GREATER_EQUAL = auto(),
  LOWER_EQUAL = auto()

class ExpectedContour(ABC):
  @abstractmethod
  def position_restrictions(
    self
  ) -> List[List[List[int, int], List[int, int], AllowedLineSideBasedOnY]]:
    pass
  
  @abstractmethod
  def shape_restrictions(self, contours: list) -> List[bool, int]:
    pass
