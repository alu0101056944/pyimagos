'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Main for visualizing the tests present in
/test/contour_operations/join_test.py (normals)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.contour_operations.join_contour import JoinContour

def prepare_image_showing_normal(contours_a, contours_b, image_size, title):
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


def test_join():
  contours = [
    np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
    np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
  ]
  joinOperation = JoinContour(0, 1, 1)
  contours_new = joinOperation.generate_new_contour(contours)
  prepare_image_showing_normal(contours, contours_new, 30,
                               'Two squares join (before, after)')
  
  contours = [
    np.array([[7, 7], [8, 8], [6, 8]]),
    np.array([[10, 10], [8, 10], [6, 10], [8, 12]])
  ]
  joinOperation = JoinContour(0, 1, 1)
  contours_new = joinOperation.generate_new_contour(contours)
  prepare_image_showing_normal(contours, contours_new, 30,
                               'Two positive neighbours join (before, after)')
  plt.show()

def visualize_tests_join() -> None:
  test_join()
