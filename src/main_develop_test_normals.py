'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Main for visualizing the tests present in /test/utils_test.py (normals)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.contour_operations.utils import  find_opposite_point_with_normals

def calculate_normal(contour, point_id, image_width, image_height):
  start_point = contour[point_id]
  num_points = len(contour)

  point_prev_idx = (point_id - 1) % num_points
  point_next_idx = (point_id + 1) % num_points
  point_prev = contour[point_prev_idx]
  point_next = contour[point_next_idx]
  tangent = point_next - point_prev
  normal = np.array([-tangent[1], tangent[0]]) # Rotate 90 degrees

  normal = normal / np.linalg.norm(normal) # normalization

  maximum_distance = ((image_width ** 2) + (image_height ** 2)) ** 0.5
  normal_projection_distance = maximum_distance

  line_start = start_point
  line_end_1 = start_point + normal * normal_projection_distance
  line_end_2 = start_point - normal * normal_projection_distance

  line_start = line_start.astype(np.int32)
  line_end_1 = line_end_1.astype(np.int32)
  line_end_2 = line_end_2.astype(np.int32)

  return line_start, line_end_1, line_start, line_end_2


def prepare_image_showing_normal(image_size, contours, contour_id, point_id,
                                 title):
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    point_color = np.array((155, 155, 155), dtype=np.uint8)
    for contour in contours:
      for point in contour:
        x, y = point
        image[y, x] = point_color

    start_point_color = np.array((0, 255, 0), dtype=np.uint8)
    x1, y1 = contours[contour_id][point_id]
    image[y1, x1] = start_point_color

    opposite_point_index = find_opposite_point_with_normals(
      contour,
      point_id,
      image_size,
      image_size
    )

    without_normal_highlighted = np.copy(image)

    opposite_color = np.array((255, 0, 0), dtype=np.uint8)
    x2, y2 = contours[0][opposite_point_index]
    image[y2, x2] = opposite_color

    with_normal_line = np.copy(without_normal_highlighted)

    normal_1_pa, normal_1_pb, normal_2_pa, normal_2_pb = calculate_normal(
      contours[contour_id],
      point_id,
      image_size,
      image_size,
    )

    cv.line(with_normal_line, normal_1_pa, normal_1_pb, (150, 0, 0), 1)
    cv.line(with_normal_line, normal_2_pa, normal_2_pb, (255, 0, 0), 1)

    concatenated = np.concatenate(
      (without_normal_highlighted, image, with_normal_line),
      axis=1
    )

    fig = plt.figure()
    plt.imshow(concatenated)
    plt.title(title)
    plt.axis('off')
    fig.canvas.manager.set_window_title(title)

def test_normals_square():
  contours = [
    np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
  ]
  prepare_image_showing_normal(30, contours, 0, 0, 'Test normal square ' \
                               '(green=start, red=opposite)')

  contours = [
      np.array([[5, 5], [10, 15], [15, 5]]),
  ]
  prepare_image_showing_normal(30, contours, 0, 0, 'Test normal triangle ' \
                              '(green=start, red=opposite)')
  
  contours = [
      np.array([[5,5], [10, 3], [13, 8], [5, 12], [1, 10], [1, 8]]),
  ]
  prepare_image_showing_normal(20, contours, 0, 0, 'Test normal concave ' \
                              '(green=start, red=opposite)')

  # Circle
  radius = 7
  center = (15, 15)
  num_points = 30
  angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
  points = []
  for angle in angles:
    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] + radius * np.sin(angle))
    points.append([x,y])
  contours = [
      np.array(points)
  ]
  prepare_image_showing_normal(30, contours, 0, 0, 'Test normal circle ' \
                            '(green=start, red=opposite)')

  plt.show()

def visualize_tests_normals() -> None:
  test_normals_square()
