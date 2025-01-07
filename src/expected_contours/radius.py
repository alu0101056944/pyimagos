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

class ExpectedContourRadius(ExpectedContour):

  def __init__(self):
    self.contour = None
    self.is_last = None
    self.top_left_corner = None
    self.top_right_corner = None
    self.bottom_right_corner = None
    self.bottom_left_corner = None
    self.image_width = None
    self.image_height = None
    self.is_last_in_branch = None


  def prepare(self, contour: list, image_width: int, image_height: int,
              is_last_in_branch: bool = False) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.image_width = image_width
    self.image_height = image_height
    self.is_last_in_branch = is_last_in_branch

    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    bounding_rect_contour = cv.boxPoints(rect)
    bounding_rect_contour = np.int32(bounding_rect_contour) # to int

    self.top_left_corner, i = get_top_left_corner(
      bounding_rect_contour,
      self.image_width,
      self.image_height
    ).tolist()

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
    x_coords = self.contour[:, 0]
    width = x_coords[np.argmax(x_coords)] - x_coords[np.argmin(x_coords)] 
    right_bound = width
    ERROR_PADDING = 10
    return [
      [
        self.top_right_corner - [0, ERROR_PADDING],
        self.top_left_corner - [0, ERROR_PADDING],
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],
      [
        self.bottom_right_corner,
        self.top_right_corner,
        AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
      ],
      [
        self.bottom_right_corner + [right_bound, 0],
        self.top_right_corner + [right_bound, 0],
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],

    ]

  def shape_restrictions(self) -> list:
    return [False, -1] if  cv.contourArea(self.contour) < 100 else [True, 0]
