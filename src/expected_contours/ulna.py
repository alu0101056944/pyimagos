'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the ulna hand bone.
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
    self.image_width = None
    self.image_height = None
    self.min_area_rect = None
    self._aspect_ratio = None
    self.reference_hu_moments = np.array(
      [
        -0.44278002,
        -1.01805892,
        -2.60498364,
        -3.04648468,
        -5.87338985,
        -3.55675893,
        -7.0069017,
      ],
      dtype=np.float64
    )
    self.orientation_line = None
    self.direction_right = None
    self.direction_left = None
    self.direction_top = None
    self.direction_bottom = None

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

    bottom_midpoint = (
      (self.bottom_left_corner[0] + self.bottom_right_corner[0]) // 2,
      (self.bottom_left_corner[1] + self.bottom_right_corner[1]) // 2
    )

    moments = cv.moments(self.contour)
    if moments["m00"] != 0: # Avoid division by zero
      centroid_x = int(moments["m10"] / moments["m00"])
      centroid_y = int(moments["m01"] / moments["m00"])
      centroid = (centroid_x, centroid_y)
    else:
      top_midpoint = (
        (self.top_left_corner[0] + self.top_right_corner[0]) // 2,
        (self.top_left_corner[1] + self.top_right_corner[1]) // 2
      )
      centroid = top_midpoint

    self.orientation_line = [bottom_midpoint, centroid]

    self.direction_right = (
      np.array(self.bottom_right_corner) - np.array(self.bottom_left_corner)
    )
    self.direction_right = (
      self.direction_right / np.linalg.norm(self.direction_right)
    )

    self.direction_left = (
      np.array(self.bottom_left_corner) - np.array(self.bottom_right_corner)
    )
    self.direction_left = (
      self.direction_left / np.linalg.norm(self.direction_left)
    )

    self.direction_top = (
      np.array(self.top_right_corner) - np.array(self.bottom_right_corner)
    )
    self.direction_top = self.direction_top / np.linalg.norm(self.direction_top)

    self.direction_bottom = (
      np.array(self.bottom_right_corner) - np.array(self.top_right_corner)
    )
    self.direction_bottom = self.direction_bottom / np.linalg.norm(self.direction_bottom)

  def next_contour_restrictions(self) -> list:
    width = self.min_area_rect[1][0]
    ERROR_PADDING = 3
    return [
      [
        np.array(self.top_right_corner) - 140,
        np.array(self.top_left_corner) - 140,
        [
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL,
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL,
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL,
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL,
        ]
      ],
      [
        self.orientation_line[0] + (
          self.direction_right * (width // 2 - ERROR_PADDING)
        ),
        self.orientation_line[1] + (
          self.direction_right * (width // 2 - ERROR_PADDING)
        ),
        [
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = +1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = -1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = 0
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # vertical
        ]
      ],

    ]

  def shape_restrictions(self) -> list:
    if len(self.contour) == 0:
      return float('inf')

    area = cv.contourArea(self.contour)
    if area < 250:
      return float('inf')

    if self._aspect_ratio < 2:
      return float('inf')
    
    if len(self.contour) < 3:
      return float('inf')
    
    min_rect_width = self.min_area_rect[1][0]
    min_rect_height = self.min_area_rect[1][1]
    hull = cv.convexHull(self.contour)
    solidity = (min_rect_width * min_rect_height) / (cv.contourArea(hull))
    if solidity > 1.6:
      return float('inf')
    
    try:
      hull_area = cv.contourArea(hull)
      significant_convexity_defects = 0
      hull_indices = cv.convexHull(self.contour, returnPoints=False)
      defects = cv.convexityDefects(self.contour, hull_indices)
      if defects is not None:
        for i in range(defects.shape[0]):
          start_index, end_index, farthest_point_index, distance = defects[i, 0]

          start = self.contour[start_index]
          end = self.contour[end_index]
          farthest = self.contour[farthest_point_index]

          defect_area = cv.contourArea(np.array([start, end, farthest]))

          if defect_area / hull_area > 0.005:
            significant_convexity_defects += 1

      if significant_convexity_defects != 3:
        return float('inf')

    except cv.error as e:
      error_message = str(e).lower()
      if 'not monotonous' in error_message:
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

  def branch_start_position_restrictions(self) -> list:
    '''Positional restrictions for when a branch has ended and a jump to other
      location is needed to reach the next jump. This is meant to be implemented
      by expected contours at the start of a branch, so that the bones at the end
      of a branch know where should the next expected contour of the next branch
      be. For example when jumping from metacarpal to next finger's distal phalanx
      in a top-left to bottom-right fashion (cv coords wise)'''
    return []

  def measure(self) -> dict:
    width = self.min_area_rect[1][0]
    height = self.min_area_rect[1][1]
    return {
      'ulna_width': width,
      'ulna_length': height,
    }
