'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the metacarpal hand bone's thumb sesamoid.
'''

import cv2 as cv
import numpy as np

from src.expected_contours.expected_contour import (
  ExpectedContour
)
from src.main_develop_corner_order import get_top_left_corner
from constants import CRITERIA_DICT

class ExpectedContourSesamoid(ExpectedContour):

  def __init__(self):
    self.contour = None
    self.top_left_corner = None
    self.top_right_corner = None
    self.bottom_right_corner = None
    self.bottom_left_corner = None
    self.image_width = None
    self.image_height = None
    self.ends_branchs_sequence = None
    self.min_area_rect = None
    self.reference_hu_moments = np.array(
      [
        -0.67457184,
        -1.7673018,
        -3.69992926,
        -4.51139064,
        -8.89464362,
        -5.77826324,
        -8.68793025,
      ],
      dtype=np.float64
    )

  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.image_width = image_width
    self.image_height = image_height
    if len(contour) == 0:
      self.contour = []
      return
    
    self.contour = np.reshape(contour, (-1, 2))

    x_values = self.contour[:, 0]
    y_values = self.contour[:, 1]
    min_x = int(np.min(x_values))
    min_y = int(np.min(y_values))
    max_x = int(np.max(x_values))
    max_y = int(np.max(y_values))
    if image_width < max_x - min_x:
      raise ValueError('Image width is not enough to cover the whole contour.')
    if image_height < max_y - min_y:
      raise ValueError('Image height is not enough to cover the whole contour.')

    rect = cv.minAreaRect(contour)
    self.min_area_rect = rect
    bounding_rect_contour = cv.boxPoints(rect)
    bounding_rect_contour = np.int32(bounding_rect_contour) # to int

    height = self.min_area_rect[1][1]
    width = self.min_area_rect[1][0]

    if height == 0 or width == 0:
      return float('inf')
    self._aspect_ratio = max(width, height) / min(width, height)

    self.top_left_corner, i = get_top_left_corner(
      bounding_rect_contour,
      self.image_width,
      self.image_height
    )

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

  def next_contour_restrictions(self) -> list:
    return []

  def shape_restrictions(self, criteria: dict = None,
                         decompose: bool = False) -> list:
    if criteria is None:
      criteria = CRITERIA_DICT

    if not decompose:
      if len(self.contour) == 0:
        return float('inf')
      
      min_rect_width = self.min_area_rect[1][0]
      min_rect_height = self.min_area_rect[1][1]
      hull = cv.convexHull(self.contour)
      solidity = (min_rect_width * min_rect_height) / (cv.contourArea(hull))
      if solidity > criteria['sesamoid']['solidity']:
        return float('inf')
      
      moments = cv.moments(self.contour)
      hu_moments = cv.HuMoments(moments)
      hu_moments = np.absolute(hu_moments)
      hu_moments_no_zeros = np.where( # to avoid DivideByZero
        hu_moments == 0,
        np.finfo(float).eps,
        hu_moments
      )
      hu_moments = (np.log10(hu_moments_no_zeros)).flatten()

      difference = np.linalg.norm(hu_moments - self.reference_hu_moments)
      return difference
    else:
      shape_fail_statuses = {
        'empty_contour': {
          'obtained_value': None,
          'threshold_value': None,
          'fail_status': None,
        },
        'solidity': {
          'obtained_value': None,
          'threshold_value': None,
          'fail_status': None,
        },
      }
      shape_fail_statuses['empty_contour']['fail_status'] = (
        True if len(self.contour) == 0 else False
      )
      shape_fail_statuses['empty_contour']['obtained_value'] = (
        len(self.contour)
      )
      shape_fail_statuses['empty_contour']['threshold_value'] = 0

      min_rect_width = self.min_area_rect[1][0]
      min_rect_height = self.min_area_rect[1][1]
      hull = cv.convexHull(self.contour)
      solidity = (min_rect_width * min_rect_height) / (cv.contourArea(hull))
      shape_fail_statuses['solidity']['fail_status'] = (
        True if solidity > criteria['medial']['solidity'] else False
      )
      shape_fail_statuses['solidity']['obtained_value'] = solidity
      shape_fail_statuses['solidity']['threshold_value'] = (
        criteria['medial']['solidity']
      )
      
      moments = cv.moments(self.contour)
      hu_moments = cv.HuMoments(moments)
      hu_moments = np.absolute(hu_moments)
      hu_moments_no_zeros = np.where( # to avoid DivideByZero
        hu_moments == 0,
        np.finfo(float).eps,
        hu_moments
      )
      hu_moments = (np.log10(hu_moments_no_zeros)).flatten()

      difference = np.linalg.norm(hu_moments - self.reference_hu_moments)

      return difference, shape_fail_statuses

  def branch_start_position_restrictions(self) -> list:
    '''Positional restrictions for when a branch has ended and a jump to other
      location is needed to reach the next jump. This is meant to be implemented
      by expected contours at the start of a branch, so that the bones at the end
      of a branch know where should the next expected contour of the next branch
      be. For example when jumping from metacarpal to next finger's distal phalanx
      in a top-left to bottom-right fashion (cv coords wise)'''
    return []

  def measure(self) -> dict:
    return {}
