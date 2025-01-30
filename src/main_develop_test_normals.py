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

from src.contour_operations.utils import  find_opposite_point

def create_minimal_image_from_contours(image: np.array,
                                       contours: list,
                                       padding = 0) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(max(0, np.min(x_values) - padding))
  min_y = int(max(0, np.min(y_values) - padding))
  max_x = int(min(image.shape[1], np.max(x_values) + padding))
  max_y = int(min(image.shape[0], np.max(y_values) + padding))

  roi_from_original = image[min_y:max_y + 1, min_x:max_x + 1]
  roi_from_original = np.copy(roi_from_original)

  # missing X padding correction on the left
  if np.min(x_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(x_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=1,
      dtype=np.uint8
    )
  
  # missing X padding correction on the right
  if np.max(x_values) + padding > image.shape[1]:
    missing_pixel_amount = np.absolute(np.max(x_values) + padding)
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8)
      ),
      axis=1,
      dtype=np.uint8
    )

  # missing Y padding correction on top
  if np.min(y_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(y_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=0,
      dtype=np.uint8
    )
  
  # missing Y padding correction on top
  if np.max(y_values) + padding > image.shape[0]:
    missing_pixel_amount = np.absolute(np.max(y_values) + padding)
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
      ),
      axis=0,
      dtype=np.uint8
    ) 

  corrected_contours = [
    points - np.array([[[min_x, min_y]]]) + padding for points in contours
  ]

  return roi_from_original, corrected_contours


def prepare_image_showing_normal(image_width: int, image_height: int, contours,
                                 contour_id, point_id, title,
                                 minimize_image: bool = False,
                                 draw_contours: bool = False):
  image = np.zeros((image_width, image_height, 3), dtype=np.uint8)
  contours = [np.reshape(contour, (-1, 1, 2)) for contour in contours]

  if minimize_image:
    image, corrected_contours = create_minimal_image_from_contours(
      image,
      contours
    )
    contours = corrected_contours

  separator_color = (255, 255, 255)
  separator_width = 2
  separator_column = np.full(
    (image.shape[0], separator_width, 3), separator_color, dtype=np.uint8
  )

  point_color = np.array((155, 155, 155), dtype=np.uint8)
  for contour in contours:
    for point in contour:
      x, y = point[0].astype(np.int64)
      image[y, x] = point_color

  centroid_color = np.array((120, 100, 70), dtype=np.uint8)
  global_centroid = np.mean(contours[0], axis=0).astype(np.int64)
  x3, y3 = global_centroid[0]
  image[y3, x3] = centroid_color

  start_point_color = np.array((0, 255, 0), dtype=np.uint8)
  x1, y1 = contours[contour_id][point_id][0].astype(np.int64)
  image[y1, x1] = start_point_color

  opposite_point_index = find_opposite_point(
    contours[0],
    point_id,
    image_width,
    image_height
  )
  if opposite_point_index is not None:

    without_normal_highlighted = np.copy(image)

    opposite_color = np.array((255, 0, 0), dtype=np.uint8)
    x2, y2 = contours[0][opposite_point_index][0].astype(np.int64)
    image[y2, x2] = opposite_color

    if draw_contours:
      image_with_drawn_contours = np.copy(image)
      cv.drawContours(image_with_drawn_contours,
                      [contour.astype(np.int64) for contour in contours],
                      -1, (0, 0, 255), 1)

      image_with_drawn_contours[y1, x1] = start_point_color
      image_with_drawn_contours[y2, x2] = opposite_color

      concatenated = np.concatenate(
        (
          without_normal_highlighted,
          separator_column,
          image,
          separator_column,
          image_with_drawn_contours
        ),
        axis=1
      )
    else:
      concatenated = np.concatenate(
        (
          without_normal_highlighted,
          separator_column,
          image
         ),
        axis=1
      )

    fig = plt.figure()
    plt.imshow(concatenated)
    plt.title(title)
    plt.axis('off')
    fig.canvas.manager.set_window_title(title)
  else:
    image_with_drawn_contours = np.copy(image)
    cv.drawContours(image_with_drawn_contours,
                    [contour.astype(np.int64) for contour in contours],
                    -1, (0, 0, 255), 1)

    image_with_drawn_contours[y1, x1] = start_point_color

    concatenated = np.concatenate(
      (
        image,
        separator_column,
        image_with_drawn_contours
      ),
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
  prepare_image_showing_normal(30, 30, contours, 0, 0, 'Test normal square ' \
                               '(green=start, red=opposite)', draw_contours=True)

  contours = [
      np.array([[5, 5], [10, 15], [15, 5]]),
  ]
  prepare_image_showing_normal(30, 30, contours, 0, 0, 'Test normal triangle ' \
                              '(green=start, red=opposite)', draw_contours=True)
  
  contours = [
      np.array([[5,5], [10, 3], [13, 8], [5, 12], [1, 10], [1, 8]]),
  ]
  prepare_image_showing_normal(30, 20, contours, 0, 0, 'Test normal concave ' \
                              '(green=start, red=opposite)', draw_contours=True)

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
  prepare_image_showing_normal(30, 30, contours, 0, 0, 'Test normal circle ' \
                            '(green=start, red=opposite)', draw_contours=True)
  
  contours = [
    np.array(
      [
        [165,   0],
        [167,   0],
        [168,   1],
        [167,   2],
        [168,   1],
        [169,   1],
        [170,   0]
      ],
    )
  ]
  image_width = 415
  image_height = 445
  prepare_image_showing_normal(image_width, image_height, contours, 0, 0,
                               'Test isolated start.', minimize_image=True,
                               draw_contours=True)
  
  contour = np.array(
    [[184, 365],
    [184, 366],
    [191, 366],
    [191, 365]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test two parallel lines.', minimize_image=True,
                               draw_contours=True)

  contour = np.array(
    [[239, 168],
    [238, 169],
    [239, 170],
    [239, 171],
    [239, 170],
    [238, 169]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test line with direction change.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [[238, 167],
    [239, 166],
    [238, 165],
    [238, 164],
    [238, 165],
    [239, 166]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test line with direction change.' \
                                'Reversed and inversed.', minimize_image=True,
                                draw_contours=True)
  
  contour = np.array(
    [[120, 296],
    [120, 303],
    [119, 304],
    [120, 303],
    [120, 301],
    [123, 298],
    [123, 296]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test kite shape', minimize_image=True,
                               draw_contours=True)
  
  contour = np.array([
    # Outer part of the C
    [20, 20], [20, 40], [30, 50],
    [50, 50], [60, 40], [60, 20],
    [50, 10], [30, 10],
    # Inner part of the C (hole)
    [24, 24], [24, 36], [32, 44],
    [44, 44], [52, 36], [52, 24],
    [44, 16], [32, 16],
  ])
  image_width = 300
  image_height = 300
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test C shape', minimize_image=True,
                               draw_contours=True)
  
  contour = np.array(
    [
      [3, 3],
      [0, 4],
      [0, 6],
      [3, 6],
      [5, 7],
      [3, 8],
      [0, 9],
      [0, 11],
      [3, 12],
      [9, 7],
      [3, 3]
    ]
  )
  image_size = 15
  prepare_image_showing_normal(image_size, image_size, [contour], 0, 5,
                               'Test folded', minimize_image=True,
                               draw_contours=True)
  
  contour = np.array(
    [
      [50, 50],
      [60, 50],
      [60, 60],
      [70, 60],
      [70, 50],
      [80, 50],
      [80, 80],
      [50, 80]
    ]
  )
  image_width = 150
  image_height = 150
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test only external intersections.',
                               minimize_image=True, draw_contours=True)

  contour = np.array(
    [
      [0, 15],
      [3, 12],
      [5, 10],
      [7, 12],
      [10, 15],
      [0, 17],
      [3, 14],
      [5, 12],
      [7, 14],
      [10, 17],
    ]
  )
  image_width = 400
  image_height = 400
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test concave fail shape.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [
      [100, 150],
      [150, 100],
      [200, 100],
      [250, 150],
      [200, 200],
      [150, 200]
    ]
  )
  image_width = 400
  image_height = 400
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test complex contour shape.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [[239, 168],
    [237, 169],
    [239, 170],
    [239, 171],
    [239, 170],
    [237, 169]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test line with direction change, 2.',
                               minimize_image=True, draw_contours=True)

  contour = np.array(
    [[239, 168],
    [237, 169],
    [238, 170],
    [239, 171],
    [239, 170],
    [237, 169]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 1,
                               'Test line with direction change, 3.',
                               minimize_image=True, draw_contours=True)

  contour = np.array(
    [[239, 168],
    [238, 168],
    [237, 169],
    [238, 170],
    [239, 171],
    [239, 170],
    [237, 169]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test line with direction change, 4.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [[239, 168],
    [238, 168],
    [237, 169],
    [238, 170],
    [239, 171],
    [240, 171],
    [240, 170],
    [240, 171]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test line with direction change, 5.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [[239, 168],
    [238, 168],
    [237, 169],
    [238, 170],
    [239, 171],
    [240, 171],
    [240, 170],
    [240, 171],
    [239, 171]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test line with direction change, 6.',
                               minimize_image=True, draw_contours=True)
  
  contour = np.array(
    [[239, 168],
    [238, 168],
    [237, 169],
    [238, 170],
    [239, 171],
    [240, 171],
    [240, 170],
    [240, 171],
    [238, 170],
    [237, 169]]
  )
  image_width = 415
  image_height = 416
  prepare_image_showing_normal(image_width, image_height, [contour], 0, 2,
                               'Test line with direction change, 7.',
                               minimize_image=True, draw_contours=True)
  plt.show()

def visualize_tests_normals() -> None:
  test_normals_square()
