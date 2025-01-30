'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for metacarpal
'''

from PIL import Image

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def create_minimal_image_from_contours(image: np.array,
                                       contours: list,
                                       padding = 5) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(max(0, np.min(x_values))) - padding
  min_y = int(max(0, np.min(y_values))) - padding
  max_x = int(min(image.shape[1], np.max(x_values))) + padding
  max_y = int(min(image.shape[0], np.max(y_values))) + padding

  roi_from_original = image[min_y:max_y + 1, min_x:max_x + 1]
  roi_from_original = np.copy(roi_from_original)

  corrected_contours = [
    points - np.array([[[min_x, min_y]]]) for points in contours
  ]

  return roi_from_original, corrected_contours

def prepare_image_showing_shape(contours, approximated_contour, image, title):

  # Third, Approximated image
  approximated_image = np.zeros((image.shape[0], image.shape[1], 3),
                                dtype=np.uint8)
  cv.drawContours(approximated_image, [approximated_contour], 0, (200, 200, 0),
                  1)

  # Separator
  separator_color = (255, 255, 255)
  separator_width = 2
  separator_column = np.full(
    (image.shape[0], separator_width, 3), separator_color, dtype=np.uint8
  )

  # First, blank image
  blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
  # Draw contours on blank image
  for i, contour in enumerate(contours):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(blank_image, contours, i, color, 1)

  # Fourth, minimum bounding box
  bounding_box_image = image.copy()
  contour = np.reshape(contours[0], (-1, 2))
  rect = cv.minAreaRect(contour)
  bounding_rect_contour = cv.boxPoints(rect)
  bounding_rect_contour = np.int32(bounding_rect_contour) # to int
  cv.drawContours(bounding_box_image, [bounding_rect_contour], 0, (0, 255, 0),
                  1)
  
  # Fifth convex hull
  convex_hull_image = image.copy()
  hull = cv.convexHull(contours[0])
  cv.drawContours(convex_hull_image, [hull], 0, (0, 255, 0), 2)

  # solidity calculation
  # contour = np.reshape(contours[0], (-1, 2))
  # rect = cv.minAreaRect(contour)
  # min_rect_width = rect[1][0]
  # min_rect_height = rect[1][1]
  # solidity = (min_rect_width * min_rect_height) / (
  #     cv.contourArea(cv.convexHull(contours[0]))
  # )

  concatenated = np.concatenate(
    (
      blank_image,
      separator_column,
      image,
      separator_column,
      approximated_image,
      separator_column,
      bounding_box_image,
      separator_column,
      convex_hull_image
    ),
    axis=1
  )

  fig = plt.figure()
  plt.imshow(concatenated)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def visualize_metacarpal_shape():
  borders_detected = Image.open('docs/metacarpal.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  minimal_image, adjusted_contours = create_minimal_image_from_contours(
    thresh,
    contours,
    padding=0
  )
  contours = adjusted_contours
  minimal_image = cv.cvtColor(minimal_image, cv.COLOR_GRAY2RGB)

  # Approximate the polygon
  epsilon = 0.02 * cv.arcLength(contours[0], closed=True)
  approximated_contour = cv.approxPolyDP(contours[0], epsilon, True)
  approximated_contour = np.reshape(approximated_contour, (-1, 2))

  prepare_image_showing_shape(contours, approximated_contour, minimal_image,
                              'Metacarpal shape.')
  
  plt.show()
