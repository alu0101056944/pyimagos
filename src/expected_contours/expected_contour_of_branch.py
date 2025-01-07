'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours.

This is an expected contour that is part of a branch of many branchs. For
example the metacarpal bone, of the first finger, where the finger is the
branch.
'''

from abc import abstractmethod

from src.expected_contours.expected_contour import ExpectedContour

class ExpectedContourOfBranch(ExpectedContour):

  @abstractmethod
    def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    pass

  @abstractmethod
  def branch_start_position_restrictions(self) -> list:
    '''Method called by last in branch to get the expected '''
    pass
