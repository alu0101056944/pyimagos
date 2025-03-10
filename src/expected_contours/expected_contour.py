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
  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    pass

  @abstractmethod
  def next_contour_restrictions(self) -> list:
    pass

  @abstractmethod
  def shape_restrictions(self, criteria: dict = None,
                         decompose: bool = False) -> list:
    pass

  @abstractmethod
  def measure(self) -> dict:
    pass
