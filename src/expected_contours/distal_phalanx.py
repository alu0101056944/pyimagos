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
  ExpectedContour, AllowedLineSideBasedOnYorXOnVertical
)
from src.expected_contours.expected_contour_of_branch import (
  ExpectedContourOfBranch
)
from src.main_develop_corner_order import get_top_left_corner

class ExpectedContourDistalPhalanx(ExpectedContourOfBranch):

  def __init__(self, encounter_amount : int,
               first_encounter: ExpectedContour = None,
               first_in_branch: ExpectedContour = None,
               ends_branchs_sequence: bool = False):
    self.contour = None
    self.top_left_corner = None
    self.top_right_corner = None
    self.bottom_right_corner = None
    self.bottom_left_corner = None
    self.image_width = None
    self.image_height = None
    self.ends_branchs_sequence = None
    self.encounter_amount = encounter_amount
    self.first_encounter = first_encounter
    self.first_in_branch = first_in_branch
    self.ends_branchs_sequence = ends_branchs_sequence
    self.approximated_contour = None
    self._aspect_ratio = None
    self.reference_hu_moments = np.array(
      [
        -0.59893488,
        -1.62052591,
        -2.46926287,
        -3.46397177,
        -6.4447155,
        -4.28778216,
        -7.03097531
      ],
      dtype=np.float64
    )

  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.image_width = image_width
    self.image_height = image_height

    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    self.min_area_rect = rect
    bounding_rect_contour = cv.boxPoints(rect)
    bounding_rect_contour = np.int32(bounding_rect_contour) # to int

    height = self.min_area_rect[1][1]
    width = self.min_area_rect[1][0]
    self._aspect_ratio = max(width, height) / min(width, height)

    self.top_left_corner, i = get_top_left_corner(
      bounding_rect_contour,
      self.image_width,
      self.image_height
    )

    # TODO experiment with the epsilon paremeter
    # Calculate perimeter of curve on the contour. Iterates all lines in contour
    # and sums the distances. closed so that it calculates distance from last to
    # first point. epsilon is distance from original curve and the approximated.
    # Changing epsilon varies how much the approx polygon is close to the original.
    epsilon = 0.02 * cv.arcLength(self.contour, closed=True)
    approximated_contour = cv.approxPolyDP(self.contour, epsilon, True)
    self.approximated_contour = np.reshape(approximated_contour, (-1, 2))

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
    ERROR_PADDING = 4
    height = self.min_area_rect[1][1]
    bottom_bound = height * 4

    return [
      [
        self.bottom_right_corner + np.array([ERROR_PADDING, 0]),
        self.top_right_corner + np.array([ERROR_PADDING, 0]),
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],
      [
        self.bottom_left_corner - np.array([ERROR_PADDING, 0]),
        self.top_left_corner - np.array([ERROR_PADDING, 0]),
        AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
      ],
      [
        self.bottom_left_corner + np.array([0, ERROR_PADDING]),
        self.bottom_right_corner + np.array([0, ERROR_PADDING]),
        AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
      ],
      [
        self.bottom_left_corner + np.array([0, bottom_bound]),
        self.bottom_right_corner + np.array([0, bottom_bound]),
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],
    ]

  def shape_restrictions(self) -> list:
    area = cv.contourArea(self.contour)
    if area <= 80:
      return [False, -1]
    
    if self._aspect_ratio < 1.3:
      return [False, -1]
    
    if self.encounter_amount > 1:
      first_encounter_aspect_ratio = self.first_encounter._aspect_ratio
      TOLERANCE = 0.3
      if abs(first_encounter_aspect_ratio - self._aspect_ratio) > TOLERANCE:
        return [False, -1]

    if len(self.approximated_contour) < 3:
      return [False, -1]
    
    min_rect_width = self.min_area_rect[1][0]
    min_rect_height = self.min_area_rect[1][1]
    hull = cv.convexHull(self.contour)
    solidity = (min_rect_width * min_rect_height) / (cv.contourArea(hull))
    if solidity > 1.3:
      return [False, -1]
    
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

        if defect_area / hull_area > 0.1:
          significant_convexity_defects += 1

    if significant_convexity_defects != 2:
      return [False, -1]

    if self.encounter_amount != 1: # little finger
      pass

    moments = cv.moments(self.contour)
    hu_moments = cv.HuMoments(moments)
    hu_moments = (np.log10(np.absolute(hu_moments))).flatten()

    difference = np.linalg.norm(hu_moments - self.reference_hu_moments)
    return [True, difference]

  def branch_start_position_restrictions(self) -> list:
    '''Positional restrictions for when a branch has ended and a jump to other
      location is needed to reach the next jump. This is meant to be implemented
      by expected contours at the start of a branch, so that the bones at the end
      of a branch know where should the next expected contour of the next branch
      be. For example when jumping from metacarpal to next finger's distal phalanx
      in a top-left to bottom-right fashion (cv coords wise)'''

    height = self.min_area_rect[1][1] 
    upper_bound = height * 4
    lower_bound = height * 2

    width = self.min_area_rect[1][0]
    right_bound = width * 5

    return [
      [
        self.top_left_corner - np.array([0, upper_bound]),
        self.top_right_corner - np.array([0, upper_bound]),
        AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL 
      ],
      [
        self.bottom_left_corner + np.array([0, lower_bound]),
        self.bottom_right_corner + np.array([0, lower_bound]),
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],
      [
        self.bottom_right_corner,
        self.top_right_corner,
        AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
      ],
      [
        self.bottom_right_corner + np.array([right_bound, 0]),
        self.top_right_corner + np.array([right_bound, 0]),
        AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
      ],
    ]

  def measure(self) -> dict:
    return {}
