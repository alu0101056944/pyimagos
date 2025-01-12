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
from src.contour_operations.utils import find_opposite_point

def create_minimal_image_from_contours(image: np.array,
                                       contours: list) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(max(0, np.min(x_values)))
  min_y = int(max(0, np.min(y_values)))
  max_x = int(min(image.shape[1], np.max(x_values)))
  max_y = int(min(image.shape[1], np.max(y_values)))

  roi_from_original = image[min_y:max_y + 1, min_x:max_x + 1]
  roi_from_original = np.copy(roi_from_original)

  corrected_contours = [
    points - np.array([[[min_x, min_y]]]) for points in contours
  ]

  return roi_from_original, corrected_contours

def prepare_image_showing_extend(contours_a, contours_b, contour_a_index,
                                 contour_b_index, image_width, image_height,
                                 title, minimize_image: bool = True):
    
  image = np.zeros((image_width, image_height, 3), dtype=np.uint8)
  

  contours_a_reshaped = [np.reshape(contour, (-1, 1, 2)) for contour in contours_a]
  contours_b_reshaped = [np.reshape(contour, (-1, 1, 2)) for contour in contours_b]

  all_contours = contours_a_reshaped + contours_b_reshaped

  if minimize_image:
    image, all_contours = create_minimal_image_from_contours(
      image,
      all_contours
    )
    contours_a_reshaped = all_contours[:len(contours_a_reshaped)]
    contours_b_reshaped = all_contours[len(contours_a_reshaped):]

  image_after_operation = np.copy(image)

  for i, contour in enumerate(contours_a_reshaped):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(image, contours_a_reshaped, i, color, 1)
  
  for i, contour in enumerate(contours_b_reshaped):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(image_after_operation, contours_b_reshaped, i, color, 1)

  point_color = np.array((155, 155, 155), dtype=np.uint8)
  for contour in contours_a_reshaped:
    for point in contour:
      x, y = point[0].astype(np.int64)
      image[y, x] = point_color

  point_color = np.array((155, 155, 155), dtype=np.uint8)
  for contour in contours_b_reshaped:
    for point in contour:
      x, y = point[0].astype(np.int64)
      image_after_operation[y, x] = point_color

  # Find closest pair

  
  fixed_contour_a = np.reshape(contours_a_reshaped[contour_a_index], (-1, 2))
  fixed_contour_b = np.reshape(contours_a_reshaped[contour_b_index], (-1, 2))
  if len(fixed_contour_a) > 0 and len(fixed_contour_b) > 0:
    # calculate closest in contour a
    min_distance = float('inf')
    closest_index_a = None
    closest_index_b = None
    for i, point_a in enumerate(fixed_contour_a):
      distances = np.sqrt(np.sum((fixed_contour_b - point_a) ** 2, axis=1))
      min_dist_local = np.min(distances)
      min_index_local = np.argmin(distances)
      if min_dist_local < min_distance:
        min_distance = min_dist_local
        closest_index_a = i
        closest_index_b = min_index_local

    closest_point_a_color = np.array((0, 255, 0), dtype=np.uint8)
    x1, y1 = fixed_contour_a[closest_index_a].astype(np.int64)
    image[y1, x1] = closest_point_a_color

    closest_point_b_color = np.array((106, 13, 173), dtype=np.uint8)
    x2, y2 = fixed_contour_b[closest_index_b].astype(np.int64)
    image[y2, x2] = closest_point_b_color

    opposite_point_index = find_opposite_point(
      fixed_contour_b,
      closest_index_b,
      image_width,
      image_height
    )

    if opposite_point_index is not None:
      opposite_color = np.array((255, 0, 0), dtype=np.uint8)
      x3, y3 = fixed_contour_b[opposite_point_index].astype(np.int64)
      image[y3, x3] = opposite_color

    separator_color = (255, 255, 255)
    separator_width = 2
    separator_column = np.full(
      (image.shape[0], separator_width, 3), separator_color, dtype=np.uint8
    )

    concatenated = np.concatenate(
      (image, separator_column, image_after_operation),
      axis=1
    )
        
    fig = plt.figure()
    plt.imshow(concatenated)
    plt.title(title)
    plt.axis('off')
    fig.canvas.manager.set_window_title(title)
  else:
    fig = plt.figure()
    plt.imshow(image)
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
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                               'extend square into rectangle ' \
                                '(before, after)')

  contours = [
    np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
    np.array([])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                              'extend square into empty ' \
                              '(before, after)')

  contours = [
    np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
    np.array([[16, 4]])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
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
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                               'extend square into two points ' \
                                '(before, after)')

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
  invasion_count = 2
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                              'extend square into rectangle invasion count 2 ' \
                              '(before, after)')

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
  invasion_count = 3
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                               'extend square into rectangle invasion count 3 ' \
                               '(before, after)')

  contours = [
    np.array([
      [4, 12],
      [4, 16],
      [8, 16],
      [16, 16],
      [16, 20],
      [20, 20],
      [20, 16],
      [20, 12],
      [16, 12],
      [16, 16],
      [8, 16],
      [8, 12],
    ]),
    np.array([[16,  4], [16,  8], [16, 24], [20, 24], [20,  8], [20,  4]])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                              'extend already invaded' \
                              '(before, after)')
  
  contours = [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([
        [16, 12],
        [16, 8],
        [20, 8],
        [20, 4],
        [16, 4],
        [16, 0],
        [20, 0],
        [24, 4],
        [24, 8],
        [24, 12],
        [20, 12],
      ])
  ]
  invasion_count = 1
  extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
  contours_new = extendOperation.generate_new_contour(contours)
  prepare_image_showing_extend(contours, contours_new, 0, 1, 30, 30,
                              'extend into concave' \
                              '(before, after)')
  
  plt.show()

def visualize_tests_extend() -> None:
  test_extend()
