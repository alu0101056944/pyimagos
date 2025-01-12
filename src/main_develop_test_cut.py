'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Main for visualizing the tests present in
/test/contour_operations/cut_test.py (normals)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.contour_operations.cut_contour import CutContour

def prepare_image_showing_cut(contours_a, contours_b, image_size, title):
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


def test_cut():
  contours = [
    np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
    np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
  ]
  joinOperation = CutContour(0, 1, 30, 30)
  contours_new = joinOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                               'Cut top left edge of left square ' \
                                '(before, after)')
  
  contours = [
    np.array([[5, 5], [10, 5], [10, 10], [7,7], [5, 10]])
  ]
  cutOperation = CutContour(0, 1, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut concave contour. (before, after)')

  contours = [
    np.array([
        [5, 5],
        [7, 8],
        [9, 8],
        [12, 9],
        [14, 8],
        [16, 7],
        [18, 7],
        [21, 6],
        [21, 5],
        [20, 4],
        [19, 4],
        [16, 4],
        [13, 3],
        [10, 3],
        [9, 2],
        [7, 4],
        [6, 5],
      ]
    )
  ]
  cutOperation = CutContour(0, 6, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut long contour. (before, after)')
  
  contours = [
    np.array([
        [5, 5],
        [7, 8],
        [9, 8],
        [12, 9],
        [14, 8],
        [16, 7],
        [18, 7],
        [21, 6],
        [21, 5],
        [20, 4],
        [19, 4],
        [16, 4],
        [13, 3],
        [10, 3],
        [9, 2],
        [7, 4],
        [6, 5],
      ]
    )
  ]
  cutOperation = CutContour(0, 9, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut long contour 2. (before, after)')
  
  contours = [
    np.array([
        [5, 5],
        [7, 8],
        [9, 8],
        [12, 9],
        [14, 8],
        [16, 7],
        [18, 7],
        [21, 6],
        [21, 5],
        [20, 4],
        [19, 4],
        [16, 4],
        [13, 3],
        [10, 3],
        [9, 2],
        [7, 4],
        [6, 5],
      ]
    )
  ]
  cutOperation = CutContour(0, 3, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut long contour 3. (before, after)')
  
  contours = [
    np.array([
        [5, 5],
        [7, 8],
        [9, 8],
        [12, 9],
        [14, 8],
        [16, 7],
        [18, 7],
        [21, 6],
        [21, 5],
        [20, 4],
        [19, 4],
        [16, 4],
        [13, 3],
        [10, 3],
        [9, 2],
        [7, 4],
        [6, 5],
      ]
    )
  ]
  cutOperation = CutContour(0, 1, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut long contour 4. (before, after)')
  
  contours = [
    np.array([[10, 10], [20, 10], [20, 20], [15, 15], [10, 20]])
  ]
  cutOperation = CutContour(0, 0, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Complex contour. (before, after)')

  contours = [
     np.array([[1, 1],[1, 29],[29, 29],[29, 1]])
  ]
  cutOperation = CutContour(0, 0, 30, 30)
  contours_new = cutOperation.generate_new_contour(contours)
  prepare_image_showing_cut(contours, contours_new, 30,
                            'Cut out of bounds. (before, after)')

  plt.show()

def visualize_tests_cut() -> None:
  test_cut()
