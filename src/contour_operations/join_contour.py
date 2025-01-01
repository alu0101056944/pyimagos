'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Penetrate contour b from closest point in contour a without reaching the
opposite side, so that the whole contour b is added to contour a
'''

import numpy as np

from src.contour_operations.contour_operation import ContourOperation

class JoinContour(ContourOperation):
  def __init__(self, contour_id1: int, contour_id2: int, invasion_count: int,
               projection_distance: int):
    self.contour_id1 = contour_id1
    self.contour_id2 = contour_id2
    self.invasion_count = invasion_count
    self.projection_distance = projection_distance

  def _find_closest_pair(contour_a: list, contour_b: list):
    overall_min_distance = float('inf')
    overall_closest_pair = None

    for point_a, i in enumerate(contour_a):
      distances = np.sqrt(np.sum((contour_b - point_a) ** 2, axis=1))

      sorted_indices = np.argsort(distances)

      min_distance_local = distances[sorted_indices[0]]
      min_distance_index = sorted_indices[0]
      second_min_distance_index = sorted_indices[1] if len(distances) > 1 else None

      if min_distance_local < overall_min_distance:
        overall_min_distance = min_distance_local
        overall_closest_pair = (i, min_distance_index, second_min_distance_index)
    
    if overall_closest_pair is not None:
      index_a = overall_closest_pair[0]
      min_index = overall_closest_pair[1]
      second_min_index = overall_closest_pair[2]
      return index_a, min_index, second_min_index
    else:
      return None
    
  def _segments_intersect(self, seg1_p1, seg1_p2, seg2_p1, seg2_p2):
    """Checks if two line segments intersect."""
    x1, y1 = seg1_p1
    x2, y2 = seg1_p2
    x3, y3 = seg2_p1
    x4, y4 = seg2_p2

    denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)

    if denom == 0: # Parallel
        return False

    t_num = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
    u_num = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

    t = t_num / denom
    u = u_num / denom
    return 0 <= t <= 1 and 0 <= u <= 1

  def generate_new_contour(self, contours: list) -> list:
    contours = np.copy(contours)

    contour_a = contours[self.contour_id1]
    contour_b = contours[self.contour_id2]
    fixed_contour_a = np.reshape(contour_a, (-1, 2))
    fixed_contour_b = np.reshape(contour_b, (-1, 2))

    if len(contour_b) < 1:
      return contours

    index_a, closest_index, second_index = self._find_closest_pair(
      fixed_contour_a,
      fixed_contour_b
    )

    if len(contour_b) <= 3:
      # Three points is just short of minimum 4 to be able to calculate the normal
      # because the normal needs the start point and next point and previous
      # point and because those are exactly three points then the normal will
      # not intersect
      return np.insert(contour_a, index_a + 1, contour_b)

    # Calculate contour b neighbor such that the connection between contour a
    # and b is straight and not crossed. Also register direction because
    # at the time of inserting contour b into a if the neighbor b direction
    # is negative (-1) then to be able to insert first index as last point
    # to visit the contour b array in the correct order then it must be
    # inserted from neighbour b to 0 first, then from len() to closest point.
    # Or From neighbor b to len() and from 0 to closest point if direction is
    # positive. Because np.insert() adds least significant index as most
    # significant index by default (axis=None).
    neighbor_a_index = (index_a - 1) % len(contour_a)
    neighbor_b_index = (closest_index - 1) % len(contour_b)
    x1, y1 = fixed_contour_a[neighbor_a_index]
    x2, y2 = fixed_contour_b[neighbor_b_index]
    intersection = self._segments_intersect(x1, y1, x2, y2)
    is_positive_direction = closest_index - 1 < 0
    if intersection:
      neighbor_b_index = (closest_index + 1) % len(contour_b)
      is_positive_direction = closest_index < len(contour_b)
    
    # If neighbor b direction is positive then 
    if is_positive_direction:
      contours[self.contour_id1] = np.insert(
        contours[self.contour_id1],
        index_a - 1,
        (
          contours[self.contour_id2][closest_index + 1:],
          contours[self.contour_id2][:closest_index + 1]
        )
      )
    else:
      contours[self.contour_id1] = np.insert(
        contours[self.contour_id1],
        index_a - 1,
        (
          contours[self.contour_id2][0:closest_index:-1],
          contours[self.contour_id2][:closest_index + 1:-1]
        )
      )

    return contours
