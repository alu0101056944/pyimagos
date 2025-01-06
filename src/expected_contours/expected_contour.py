'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours.

The AllowedLineSideBasedOnYorXOnVertical enum decides which side of the two sides
that the line has is the space where contours relative to this
expected contour are allowed to be.
'''

from abc import ABC, abstractmethod
from enum import Enum, auto

class AllowedLineSideBasedOnYorXOnVertical(Enum):
  GREATER = auto(),
  LOWER = auto(),
  GREATER_EQUAL = auto(),
  LOWER_EQUAL = auto()

class ExpectedContour(ABC):

  @abstractmethod
  def prepare(self) -> None:
    pass

  @abstractmethod
  def position_restrictions(self) -> list:
    pass
  
  @abstractmethod
  def shape_restrictions(self, contours: list) -> list:
    pass

  @abstractmethod
  def get_next_to_restrictions(self) -> list:
    pass
