'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Main for visualizing the tests present in /test/utils_test.py (segment
intersection)
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.contour_operations.utils import  blend_colors_with_alpha

# TODO Somehow the opacity change is very small.
def draw_points_and_two_segments(contours, point_a, closest, neighbour_a,
                                 neighbour_b, title=''):
  '''Draws two segments and contour points with alpha blending.'''
  line_a_color = (0, 0, 255, 255)
  image_with_line_a = np.zeros((30, 30, 4), dtype=np.uint8)
  image_with_line_a[:, :, 3] = 255
  cv.line(image_with_line_a, tuple(neighbour_a), tuple(neighbour_b),
          color=line_a_color, thickness=1)

  line_b_color = (255, 0, 0, 240)
  image_with_line_b = np.zeros((30, 30, 4), dtype=np.uint8)
  image_with_line_b[:, :, 3] = 0
  cv.line(image_with_line_b, tuple(point_a), tuple(closest),
          color=line_b_color, thickness=1)
  
  for i in range(image_with_line_a.shape[0]):
    for j in range(image_with_line_a.shape[1]):
      background_color = image_with_line_a[i, j]
      foreground_color = image_with_line_b[i, j]
      blended_color, blended_alpha = blend_colors_with_alpha(
        background_color,
        foreground_color
      )
      image_with_line_a[i, j, :3] = blended_color
      image_with_line_a[i, j, 3] = blended_alpha

  for contour in contours:
    for point in contour:
      x, y = point
      background_color = image_with_line_a[y, x]
      foreground_color = np.array((155, 155, 155, 125), dtype=np.uint8)

      blended_color, blended_alpha = blend_colors_with_alpha(
        background_color,
        foreground_color
      )
      image_with_line_a[y, x, :3] = blended_color
      image_with_line_a[y, x, 3] = blended_alpha

  # image_with_line_a = cv.resize(image_with_line_a, (400, 400),
  #                               interpolation=cv.INTER_NEAREST)
  
  # Debugging fragment
  # def __to_structured(a: np.array):
  #   '''So that each row is treated as a single element so that np.unique() 
  #   works correctly'''
        
  #   colors_array = np.dtype(
  #     [('b', a.dtype),
  #     ('g', a.dtype),
  #     ('r', a.dtype),
  #     ('a', a.dtype)]
  #   )
  #   return a.view(colors_array)
  # unique_colors = [np.unique(__to_structured(row), return_counts=True) for row in image_with_line_a]

  fig = plt.figure()
  plt.imshow(image_with_line_a)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def test_intersection_square():
  contours = [
    np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
    np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
  ]

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = tuple(contours[1][1])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                               neighbour_b, title='Case a. No intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = tuple(contours[1][3])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                               neighbour_b, title='Case b. No intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][0])
  neighbour_b = tuple(contours[1][0])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                               neighbour_b, title='Case c parallel touching. ' \
                                'No intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = tuple(contours[1][0])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                               neighbour_b, title='Case d. Intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][3])
  neighbour_b = tuple(contours[1][1])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                               neighbour_b, title='Case e. Intersection.')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][3])
  neighbour_a[1] = neighbour_a[1] + 1
  neighbour_b = tuple(contours[1][1])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                                neighbour_b, title='Adjacent origin. ' \
                                  'No intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][2])
  neighbour_a[1] = neighbour_a[1] + 1
  neighbour_b = tuple(contours[1][3])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                              neighbour_b, title='No intersection due to ' \
                                'lengths. No intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][0])
  neighbour_b = list(neighbour_a)
  neighbour_b[0] = neighbour_a[0] + 2
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                              neighbour_b, title='Parallel no touching. ' \
                                'No intersection')
  
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][3])
  neighbour_a[1] = neighbour_a[1] + 1
  neighbour_b = list(contours[1][0])
  neighbour_b[1] = neighbour_b[1] + 1
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                            neighbour_b, title='Parallel adjacent.' \
                                'No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = list(contours[1][3])
  neighbour_b[1] = neighbour_b[1] + 2
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                              neighbour_b, title='No intersection due to ' \
                                'lengths. Lower. No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = list(contours[1][2])
  neighbour_b[1] = neighbour_b[1] + 6
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                            neighbour_b, title='No intersection due to ' \
                              'advanced origin b. No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[1][2]) 
  neighbour_a[1] = neighbour_a[1] + 6
  neighbour_b = tuple(contours[0][2])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                          neighbour_b, title='No intersection due to ' \
                            'advanced origin b. (reversed). No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2])
  neighbour_b = list(contours[1][2])
  neighbour_b[1] = neighbour_b[1] + 6
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                          neighbour_b, title='No intersection due to ' \
                            'advanced origin b. Inverse arguments. ' \
                              'No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[1][2]) 
  neighbour_a[1] = neighbour_a[1] + 6
  neighbour_b = tuple(contours[0][2])
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                        neighbour_b, title='No intersection due to ' \
                          'advanced origin b. Inverse arguments. Reversed. ' \
                            'No intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = tuple(contours[0][2]) 
  neighbour_b = list(contours[1][0])
  neighbour_b[0] = neighbour_b[0] - 4
  neighbour_b[1] = neighbour_b[1] - 4
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                      neighbour_b, title='Crossed intersection. ' \
                        'Intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][2]) 
  neighbour_a[0] = neighbour_a[0] + 4
  neighbour_b = list(contours[1][0])
  neighbour_b[0] = neighbour_b[0] - 4
  neighbour_b[1] = neighbour_b[1] - 4
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                    neighbour_b, title='Crossed intersection. Further on x. ' \
                      'Intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][2]) 
  neighbour_a[0] = neighbour_a[0] + 2
  neighbour_b = list(contours[1][0])
  neighbour_b[0] = neighbour_b[0] - 4
  neighbour_b[1] = neighbour_b[1] - 4
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                  neighbour_b, title='Crossed intersection. Further on x. ' \
                    ' from +4 to +2 (x). Intersection')
  
  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][2]) 
  neighbour_a[0] = neighbour_a[0] + 6
  neighbour_b = list(contours[1][0])
  neighbour_b[0] = neighbour_b[0] - 4
  neighbour_b[1] = neighbour_b[1] - 4
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
                neighbour_b, title='Crossed intersection. Further on x. ' \
                  ' from +4 to +6 (x). Intersection')

  point_a = tuple(contours[0][3])
  closest_point = tuple(contours[1][0])
  neighbour_a = list(contours[0][2]) 
  neighbour_a[0] = neighbour_a[0] + 10
  neighbour_b = list(contours[1][0])
  neighbour_b[0] = neighbour_b[0] - 4
  neighbour_b[1] = neighbour_b[1] - 4
  draw_points_and_two_segments(contours, point_a, closest_point, neighbour_a,
              neighbour_b, title='Crossed intersection. Other side x. ' \
                'Intersection')
  plt.show()

def visualize_tests() -> None:
  test_intersection_square()
