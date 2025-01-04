'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Main for visualizing the tests present in
/test/contour_operations/extend_test.py (normals)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.contour_operations.extend_contour import ExtendContour

def prepare_image_showing_extend(contours_a, contours_b, image_size, title):
  image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
  image_after_operation = np.copy(image)

  for i, contour in enumerate(contours_a):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(image, contours_a, i, color, 1)

  for i, contour in enumerate(contours_b):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(image_after_operation, contours_b, i, color, 1)

  for i, contour in enumerate(contours_a):
    color = np.array(
      [(i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256],
      dtype=np.uint8
    )
    color = color - 30
    for point in contour:
      x, y = point
      image[y, x] = color

  for i, contour in enumerate(contours_b):
    color = np.array(
      [(i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256],
      dtype=np.uint8
    )
    color = color - 30
    for point in contour:
      x, y = point
      image_after_operation[y, x] = color

  concatenated = np.concatenate(
    (image, image_after_operation),
    axis=1
  )

  fig = plt.figure()
  plt.imshow(concatenated)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)


def test_extend():
  contours = [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([
        [16, 4],
        [16, 8],
        [16, 12],
        [16, 16],
        [16, 20],
        [16, 24],
        [20, 24],
        [20, 20],
        [20, 16],
        [20, 12],
        [20, 8],
        [20, 4],
      ])
    ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 30,
                               'extend square into rectangle ' \
                                '(before, after)')

  contours = [
    np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
    np.array([])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 30,
                              'extend square into empty ' \
                              '(before, after)')

  contours = [
    np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
    np.array([[16, 4]])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 30,
                            'extend square into one point ' \
                            '(before, after)')

  contours = [
    np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
    np.array([
      [16, 4],
      [16, 8],
    ])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 30,
                               'extend square into two points ' \
                                '(before, after)')
  
  
  plt.show()

def visualize_tests_extend() -> None:
  test_extend()
