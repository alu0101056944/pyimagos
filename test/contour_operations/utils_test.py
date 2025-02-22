'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.contour_operations.utils import blend_colors_with_alpha
from src.contour_operations.utils import segments_intersect
from src.contour_operations.utils import line_segment_intersection
from src.contour_operations.utils import find_opposite_point
from src.contour_operations.utils import filter_internal_interceptions

class TestContourUtils:

  def test_black_background_white_foreground(self):
    background_color = np.array([0, 0, 0, 255], dtype=np.float32)
    foreground_color = np.array([255, 255, 255, 127.5], dtype=np.float32)
    blend_color, blend_alpha = blend_colors_with_alpha(
      background_color,
      foreground_color
    )
    assert np.array_equal(blend_color, np.array([127, 127, 127], dtype=np.uint8))
    assert np.array_equal(blend_alpha, 255)

  def test_non_blacks_get_properly_blended(self):
    background_color = np.array([255, 0, 0, 255], dtype=np.float32)
    foreground_color = np.array([0, 0, 255, 204], dtype=np.float32)
    blend_color, blend_alpha = blend_colors_with_alpha(
      background_color,
      foreground_color
    )
    assert np.array_equal(blend_color, np.array([50, 0, 204], dtype=np.uint8))
    assert np.array_equal(blend_alpha, 255)

    
  @pytest.fixture(scope='class')
  def contours(self):
    '''Two squares next to each other'''
    yield [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
    ]

  def test_intersection_square_case_a(self, contours: list) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][2])
    neighbour_b = tuple(contours[1][1])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_square_case_b(self, contours: list) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][2])
    neighbour_b = tuple(contours[1][3])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_square_case_c_parallel_touching(self,
                                                        contours: list) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][0])
    neighbour_b = tuple(contours[1][0])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_square_case_d(self, contours: list) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][2])
    neighbour_b = tuple(contours[1][0])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == True


  def test_intersection_square_case_e(self, contours: list) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][3])
    neighbour_b = tuple(contours[1][1])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == True

  def test_intersection_square_adjacent_origin(self, contours) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][3])
    neighbour_a[1] = neighbour_a[1] + 1
    neighbour_b = tuple(contours[1][1])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_no_intersection_due_to_lengths(self, contours) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2])
    neighbour_a[1] = neighbour_a[1] + 1
    neighbour_b = tuple(contours[1][3])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_no_intersection_parallel_no_touching(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][0])
    neighbour_b = list(neighbour_a)
    neighbour_b[0] = neighbour_a[0] + 2
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_intersection_parallel_adjacent(self, contours) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][3])
    neighbour_a[1] = neighbour_a[1] + 1
    neighbour_b = list(contours[1][0])
    neighbour_b[1] = neighbour_b[1] + 1
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_no_intersection_due_to_lengths_lower(
    self,
    contours
  ) -> None:
    point_a = np.array(contours[0][3], dtype=np.float32)
    closest_point = np.array(contours[1][0], dtype=np.float32)
    neighbour_a = np.array(contours[0][2], dtype=np.float32)
    neighbour_b = np.array(contours[1][3], dtype=np.float32)
    neighbour_b[1] = neighbour_b[1] + 2
    u, t = line_segment_intersection(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    no_intersection = not (u and t)
    assert no_intersection

  def test_intersection_no_intersection_due_to_advanced_origin(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2])
    neighbour_b = list(contours[1][2])
    neighbour_b[1] = neighbour_b[1] + 6
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_no_intersection_due_to_advanced_origin_reversed(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[1][2]) 
    neighbour_a[1] = neighbour_a[1] + 6
    neighbour_b = list(contours[0][2])
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b
    )
    assert intersection == False

  def test_intersection_no_intersection_due_to_advanced_origin_inverse_args(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][2])
    neighbour_b = list(contours[1][2])
    neighbour_b[1] = neighbour_b[1] + 6
    intersection, _, _ = segments_intersect(
      neighbour_a,
      neighbour_b,
      point_a,
      closest_point,
    )
    assert intersection == False

  def test_intersection_no_intersection_due_to_advanced_origin_inverse_args_reversed(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[1][2]) 
    neighbour_a[1] = neighbour_a[1] + 6
    neighbour_b = tuple(contours[0][2])
    intersection, _, _ = segments_intersect(
      neighbour_a,
      neighbour_b,
      point_a,
      closest_point,
    )
    assert intersection == False


  def test_intersection_intersection_crossed(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = tuple(contours[0][2]) 
    neighbour_b = list(contours[1][0])
    neighbour_b[0] = neighbour_b[0] - 4
    neighbour_b[1] = neighbour_b[1] - 4
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b,
    )
    assert intersection == True

  def test_intersection_intersection_crossed_further_x(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2]) 
    neighbour_a[0] = neighbour_a[0] + 4
    neighbour_b = list(contours[1][0])
    neighbour_b[0] = neighbour_b[0] - 4
    neighbour_b[1] = neighbour_b[1] - 4
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b,
    )
    assert intersection == True

  def test_intersection_intersection_crossed_further_x_2(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2]) 
    neighbour_a[0] = neighbour_a[0] + 2
    neighbour_b = list(contours[1][0])
    neighbour_b[0] = neighbour_b[0] - 4
    neighbour_b[1] = neighbour_b[1] - 4
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b,
    )
    assert intersection == True

  def test_intersection_intersection_crossed_even_further_x(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2]) 
    neighbour_a[0] = neighbour_a[0] + 6
    neighbour_b = list(contours[1][0])
    neighbour_b[0] = neighbour_b[0] - 4
    neighbour_b[1] = neighbour_b[1] - 4
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b,
    )
    assert intersection == True

  def test_intersection_intersection_crossed_other_side_x(
    self,
    contours
  ) -> None:
    point_a = tuple(contours[0][3])
    closest_point = tuple(contours[1][0])
    neighbour_a = list(contours[0][2]) 
    neighbour_a[0] = neighbour_a[0] + 10
    neighbour_b = list(contours[1][0])
    neighbour_b[0] = neighbour_b[0] - 4
    neighbour_b[1] = neighbour_b[1] - 4
    intersection, _, _ = segments_intersect(
      point_a,
      closest_point,
      neighbour_a,
      neighbour_b,
    )
    assert intersection == True

  def test_normal_one_point(self):
    contour = np.array([[15, 15]], dtype=np.int32)
    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    assert opposite_point_index == None

  def test_normal_two_point(self):
    contour = np.array([[15, 15], [20, 15]], dtype=np.int32)
    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    assert opposite_point_index == None

  def test_normal_triangle(self):
    contours = [
        np.array([[5, 5], [10, 15], [15, 5]]),
    ]
    contour = contours[0]
    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    assert opposite_point_index == 1

  def test_normal_square(self, contours):
    contour = contours[0]

    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    
    assert opposite_point_index == 2

  def test_normal_concave(self):
    contours = [
        np.array([[5,5], [10, 3], [13, 8], [5, 12], [1, 10], [1, 8]]),
    ]
    contour = contours[0]

    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    
    assert opposite_point_index == 3

  def test_normal_circle(self):
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

    contour = contours[0]

    image_size = 30
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    
    assert opposite_point_index == 15

  def test_normal_isolated_start(self):
    contour = np.array(
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

    image_width = 415
    image_height = 445
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_width,
      image_height
    )
    assert opposite_point_index == 2

  def test_normal_no_float_precision_error(self):
    contour = np.array(
      [[184, 365],
      [184, 366],
      [191, 366],
      [191, 365]]
    )
    image_width = 415
    image_height = 416
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index == 3

  def test_opposite_line_with_direction_change_case(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_opposite_line_with_direction_change_inversed_reversed_case(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_opposite_line_kite_shape(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      2,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_opposite_equal_points(self):
    contour = np.array(
      [[2, 2],
      [2, 2],
      [2, 2]]
    )
    image_size = 5
    opposite_point_index = find_opposite_point(
      contour,
      0,
      image_size,
      image_size
    )
    assert opposite_point_index == None

  def test_opposite_folded(self):
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
    image_size = 10
    opposite_point_index = find_opposite_point(
      contour,
      5,
      image_size,
      image_size
    )
    assert opposite_point_index == 8
    
  def test_only_external_intersections(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index == 0

  def test_complex_contour_opposite_point(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index == 4

  # estudiar cuando sucede el caso de forma cercana.
  def test_opposite_line_with_direction_change_case_2(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_opposite_line_with_direction_change_case_3(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      1,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_opposite_line_with_direction_change_case_4(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      2,
      image_width,
      image_height
    )
    assert opposite_point_index is None
    

  def test_opposite_line_with_direction_change_case_5(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      2,
      image_width,
      image_height
    )
    assert opposite_point_index == 6
    

  def test_opposite_line_with_direction_change_case_6(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      2,
      image_width,
      image_height
    )
    assert opposite_point_index == 3

  def test_opposite_line_with_direction_change_case_7(self):
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
    opposite_point_index = find_opposite_point(
      contour,
      2,
      image_width,
      image_height
    )
    assert opposite_point_index is None

  def test_none_input_to_filter_internal_interceptions(self):
    no_opposite_point_contour = np.array(
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
    filtered_interceptions = filter_internal_interceptions(
      None,
      2,
      no_opposite_point_contour
    )
    assert filtered_interceptions is None
