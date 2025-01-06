'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the proximal phalanx hand bone.
'''

import cv2 as cv
import numpy as np

from src.expected_contours.expected_contour import (
  ExpectedContour, AllowedLineSideBasedOnYorXOnVertical
)

from src.main_develop_corner_order import get_top_left_corner

class ExpectedContourUlna(ExpectedContour):

  def __init__(self):
    self.contour = None
    self.is_last = None
    self.top_left_corner = None
    self.top_right_corner = None
    self.bottom_right_corner = None
    self.bottom_left_corner = None

  def prepare(self, contour: list, is_last: bool = False) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.is_last = is_last
    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    bounding_rect_contour = cv.boxPoints(rect)
    bounding_rect_contour = np.int32(bounding_rect_contour) # to int

    self.top_left_corner, i = get_top_left_corner(bounding_rect_contour).tolist()
    # assumming clockwise
    self.top_right_corner = bounding_rect_contour[
      (i + 1) % len(bounding_rect_contour)
    ].tolist()
    self.bottom_right_corner = bounding_rect_contour[
      (i + 2) % len(bounding_rect_contour)
    ].tolist()
    self.bottom_left_corner = bounding_rect_contour[
      (i + 3) % len(bounding_rect_contour)
    ].tolist()

  def position_restrictions(self) -> list:
    '''Expected to be the last bone to be reached so no restrictions'''
    return []

  def shape_restrictions(self) -> list:
    return [False, -1] if  cv.contourArea(self.contour) < 100 else [True, 0]
