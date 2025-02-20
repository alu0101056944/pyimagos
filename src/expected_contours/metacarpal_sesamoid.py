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
  ExpectedContour, AllowedLineSideBasedOnYorXOnVertical
)
from src.main_develop_corner_order import get_top_left_corner

class ExpectedContourMetacarpalSesamoid(ExpectedContour):

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
    return []

  def shape_restrictions(self) -> list:
    area = cv.contourArea(self.contour)
    if area < 10:  # Minimum area for sesamoid, adjust as needed
      return float('inf')

    # Calculate perimeter of curve on the contour. Iterates all lines in contour
    # and sums the distances. closed so that it calculates distance from last to
    # first point. epsilon is distance from original curve and the approximated.
    # Changing epsilon varies how much the approx polygon is close to the original.
    epsilon = 0.02 * cv.arcLength(self.contour, closed=True)
    approximated_contour = cv.approxPolyDP(self.contour, epsilon, True)
    approximated_contour = np.reshape(approximated_contour, (-1, 2))

    if len(approximated_contour) < 3: # min 3 points to form a shape
      return float('inf')

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

    score = area / 10

    # Penalize for not being close to a round shape
    score -= (abs(average_angle - 180) * 0.1)

    # Penalty for too many or too few corners
    if len(approximated_contour) < 3 or len(approximated_contour) > 6:
      score -= 5

    if score < 1:
      return float('inf')

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
      f'metacarpal_sesamoid_{self.encounter_amount}_width': width,
      f'metacarpal_sesamoid_{self.encounter_amount}_length': height,
    }
