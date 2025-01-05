'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the distal phalanx hand bone.
'''

from typing import List

import cv2 as cv
import numpy as np

from src.expected_contours.expected_contour import (
  ExpectedContour, AllowedLineSideBasedOnY
)

class ExpectedContourDistalPhalanx(ExpectedContour):

  def initialize(self, contour: list):
    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    self.bounding_rect_contour = cv.boxPoints(rect)

  def position_restrictions(
    self
  ) -> List[List[List[int, int], List[int, int], AllowedLineSideBasedOnY]]:
    top_left_corner = self.bounding_rect_contour[0].tolist()
    top_right_corner = self.bounding_rect_contour[3].tolist()
    bottom_right_corner = self.bounding_rect_contour[2].tolist()
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

  def shape_restrictions(self) -> List[bool, int]:
    [False, -1] if  cv.contourArea(self.contour) < 100 else [True, 0]
