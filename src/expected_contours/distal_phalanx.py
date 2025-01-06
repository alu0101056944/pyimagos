'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the distal phalanx hand bone.
'''

import cv2 as cv
import numpy as np

from src.expected_contours.expected_contour import (
  ExpectedContour, AllowedLineSideBasedOnY
)

from src.main_develop_corner_order import get_top_left_corner

class ExpectedContourDistalPhalanx(ExpectedContour):

  def __init__(self, contour: list, proximal_phalanx_contour: list):
    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    self.bounding_rect_contour = cv.boxPoints(rect)
    self.bounding_rect_contour = np.int32(self.bounding_rect_contour) # to int
    self.proximal_phalanx_contour = proximal_phalanx_contour

  def position_restrictions(self) -> list:
    top_left_corner, i = get_top_left_corner(self.bounding_rect_contour).tolist()
    # assumming clockwise
    top_right_corner = self.bounding_rect_contour[
      (i + 1) % len(self.bounding_rect_contour)
    ].tolist()
    bottom_right_corner = self.bounding_rect_contour[
      (i + 2) % len(self.bounding_rect_contour)
    ].tolist()
    return [
      [
        bottom_right_corner,
        top_right_corner,
        AllowedLineSideBasedOnY.LOWER
      ],
      [
        top_left_corner,
        top_right_corner,
        AllowedLineSideBasedOnY.GREATER_EQUAL
      ]
    ]

  def shape_restrictions(self) -> list:
    return [False, -1] if  cv.contourArea(self.contour) < 100 else [True, 0]
