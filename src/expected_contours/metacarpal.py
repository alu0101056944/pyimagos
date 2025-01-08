'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the metacarpal hand bone.
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

class ExpectedContourMetacarpal(ExpectedContourOfBranch):

  def __init__(self, encounter_amount : int,
               first_occurence: ExpectedContour = None,
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
    self.first_occurence = first_occurence
    self.first_in_branch = first_in_branch
    self.ends_branchs_sequence = ends_branchs_sequence

  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.image_width = image_width
    self.image_height = image_height

    self.contour = np.reshape(contour, (-1, 2))
    rect = cv.minAreaRect(contour)
    self.min_area_rect = rect
    bounding_rect_contour = cv.boxPoints(rect)
    bounding_rect_contour = np.int32(bounding_rect_contour) # to int

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
    if not self.ends_branchs_sequence:
      return self.first_in_branch.branch_start_position_restrictions()
    else: # expect ulna
      ERROR_PADDING = 10
      width = self.min_area_rect[1][0]
      right_bound = width
      left_bound = width * 6
      return [
        [
          self.bottom_left_corner + [0, ERROR_PADDING],
          self.bottom_right_corner + [0, ERROR_PADDING],
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
        ],
        [
          self.bottom_left_corner - [left_bound, 0],
          self.top_left_corner - [left_bound, 0],
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL
        ],
        [
          self.bottom_right_corner + [right_bound, 0],
          self.top_right_corner + [right_bound, 0],
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL
        ],
      ]

  def shape_restrictions(self) -> list:
    area = cv.contourArea(self.contour)
    if area < 100:
      return [False, -1]

    epsilon = 0.02 * cv.arcLength(self.contour, closed=True)
    approximated_contour = cv.approxPolyDP(self.contour, epsilon, True)
    approximated_contour = np.reshape(approximated_contour, (-1, 2))

    if len(approximated_contour) < 3:
      return [False, -1]

    angles = []
    for i in range(len(approximated_contour)):
      p1 = approximated_contour[i - 1]
      p2 = approximated_contour[i]
      p3 = approximated_contour[(i + 1) % len(approximated_contour)]

      v1 = p1 - p2
      v2 = p3 - p2
      angle = np.degrees(
        np.arctan2(
          np.linalg.det([v1, v2]),
          np.dot(v1, v2)
        )
      )

      angles.append(angle)

    angles_abs = [abs(angle) for angle in angles]
    average_angle = sum(angles_abs) / len(angles_abs)

    curvature_score = 0
    for i in range(1, len(self.contour) - 1):
      p1 = self.contour[i - 1]
      p2 = self.contour[i]
      p3 = self.contour[i + 1]

      v1 = p1 - p2
      v2 = p3 - p2

      angle = np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
      curvature_score += abs(angle)

    if len(self.contour) > 0:
      curvature_score /= len(self.contour)
    else:
      curvature_score = 0

    score = area / 100

    if self.encounter_amount == 1: # little finger
      score -= (abs(average_angle - 180) * 0.8)
      score -= (curvature_score * 0.05)
      if len(approximated_contour) < 2 or len(approximated_contour) > 4:
        score -= 5
    elif self.encounter_amount == 2: # ring finger
      score -= (abs(average_angle - 180) * 0.8)
      score -= (curvature_score * 0.05)
      if len(approximated_contour) < 2 or len(approximated_contour) > 4:
        score -= 5
    elif self.encounter_amount == 3: # middle finger
      score -= (abs(average_angle - 180) * 0.7)
      score -= (curvature_score * 0.05)
      if len(approximated_contour) < 2 or len(approximated_contour) > 4:
        score -= 5
    elif self.encounter_amount == 4: # index finger
      score -= (abs(average_angle - 180) * 0.6)
      score -= (curvature_score * 0.05)
      if len(approximated_contour) < 2 or len(approximated_contour) > 4:
        score -= 5
    elif self.encounter_amount == 5: # thumb finger
      score -= (abs(average_angle - 180) * 0.6)
      score -= (curvature_score * 0.05)
      if len(approximated_contour) < 2 or len(approximated_contour) > 4:
        score -= 5

    if score < 1:
      return [False, -1]

    return [True, score]

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
      f'metacarpal_{self.encounter_amount}_width': width,
      f'metacarpal_{self.encounter_amount}_length': height,
    }
