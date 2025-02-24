'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Tests for finding the sesamoid near the fifth metacarpal.
'''

from PIL import Image

import numpy as np
import cv2 as cv

from src.expected_contours.sesamoid import (
  ExpectedContourSesamoid
)

from src.main_develop_contours_gui import (
  draw_contours, draw_contour_points
)

def get_bounding_box_xy(contour):
  x, y, w, h = cv.boundingRect(contour)
  return x, y

def get_contours_in_region(image, contours, region_rect, padding):
  x, y, w, h = region_rect
  region_x1 = max(0, x - padding)
  region_y1 = max(0, y - padding)
  region_x2 = min(image.shape[1], x + w + padding)
  region_y2 = min(image.shape[0], y + h + padding)

  contours_in_region = []

  for contour in contours:
    contour_x, contour_y, contour_w, contour_h = cv.boundingRect(contour)
    contour_x1 = contour_x
    contour_y1 = contour_y
    contour_x2 = contour_x + contour_w
    contour_y2 = contour_y + contour_h

    if (
      region_x1 <= contour_x1
      and region_y1 <= contour_y1
      and region_x2 >= contour_x2
      and region_y2 >= contour_y2
    ):
      contours_in_region.append(contour)

  return contours_in_region

def create_minimal_image_from_contour(image: np.array,
                                       contour: np.array,
                                       padding: int) -> np.array:
  x, y, w, h = cv.boundingRect(contour)

  x1 = max(0, x - padding)
  y1 = max(0, y - padding)
  x2 = min(image.shape[1], x + w + padding)
  y2 = min(image.shape[0], y + h + padding)

  cropped_image = image[y1:y2, x1:x2]

  return cropped_image

def case_1():
  borders_detected = Image.open('docs/metacarpal_bone_minimal_larger.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_BGR2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
  contours_drawn_image, contour_colors = draw_contour_points(
    thresh,
    contours,
    draw_points=True
  )
  contours_drawn_image = draw_contours(
    contours_drawn_image,
    contours,
    contour_colors
  )

  sorted_contours = sorted(contours, key=get_bounding_box_xy)
  focused_image = create_minimal_image_from_contour(
    contours_drawn_image,
    sorted_contours[0],
    padding=10
  )

  region_rect = cv.boundingRect(sorted_contours[0])
  contours_within_focused_image = get_contours_in_region(
    focused_image,
    contours,
    region_rect, #TODO change rect region to image and not contour
    padding=10
  )

  shape_values = []
  sesamoid_instance = ExpectedContourSesamoid()
  for contour_within in contours_within_focused_image:
    sesamoid_instance.prepare(
      contour_within,
      focused_image.shape[1],
      focused_image.shape[0]
    )
    shape_value = sesamoid_instance.shape_restrictions()
    shape_values.append(shape_value)

  sesamoid_contour_index = np.argmax(shape_values)
  sesamoid_contour = contours[sesamoid_contour_index]

  # show it

  cv.drawContours(
    focused_image,
    contours_within_focused_image,
    sesamoid_contour_index,
    color=(0, 0, 255),
    thickness=2
  )

  contours_drawn_image_scaled1 = cv.resize(contours_drawn_image, (0, 0), fx=2,
                      fy=2,
                      interpolation=cv.INTER_NEAREST)
  contours_drawn_image_scaled2 = cv.resize(focused_image, (0, 0), fx=2,
                      fy=2,
                      interpolation=cv.INTER_NEAREST)
  
  cv.imshow('Non focused in', contours_drawn_image_scaled1)
  cv.imshow('Focused in', contours_drawn_image_scaled2)
  cv.waitKey(0)
  cv.destroyAllWindows()

def find_sesamoid_main():
  case_1()
