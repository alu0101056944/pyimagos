'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the radius hand bone.
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

  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    '''This is needed to select the contour that this class will work on'''
    self.image_width = image_width
    self.image_height = image_height

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

  def next_contour_restrictions(self) -> list:
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
    score -= (abs(average_angle - 180) * 0.5)
    score -= (curvature_score * 0.05)
    if len(approximated_contour) < 2 or len(approximated_contour) > 5:
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
