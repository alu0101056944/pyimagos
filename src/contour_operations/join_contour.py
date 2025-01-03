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
from src.contour_operations.utils import segments_intersect

class JoinContour(ContourOperation):
  def __init__(self, contour_id1: int, contour_id2: int, invasion_count: int):
    self.contour_id1 = contour_id1
    self.contour_id2 = contour_id2
    self.invasion_count = invasion_count

  def _find_closest_pair(self, contour_a: list, contour_b: list):
    overall_min_distance = float('inf')
    overall_closest_pair = None

    for i, point_a in enumerate(contour_a):
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

  def generate_new_contour(self, contours: list) -> list:
    contours = [np.copy(array) for array in contours]

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

    if len(contour_b) < 3:
      # Three points is just short of minimum 3 to be able to calculate the normal
      # because the normal needs the start point and next point and previous
      # point and because those are exactly three points then the normal will
      # not intersect
      # TODO fix this, it's most likely wrong, unit test it.
      return np.insert(contour_a, index_a + 1, contour_b)

    # Calculate contour b neighbor such that the connection between contour a
    # and b is straight and not crossed. Choose neighbor direction for both
    # a and b such that it doesn't happen. The direction is important
    # for concatenating b into a while keeping the point sequence intact
    possible_directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    bridge_intersection = True
    own_contour_intersection = True
    neighbor_a_index = None
    neighbor_b_index = None
    neighbor_a_is_positive = None
    neighbor_b_is_positive = None
    for possible_direction in possible_directions:
      direction_a, direction_b = possible_direction
      if not bridge_intersection and not own_contour_intersection:
        neighbor_a_is_positive = (
          (index_a + direction_a) % len(contour_a) > index_a
        )
        neighbor_b_is_positive = (
          (closest_index + direction_b) % len(contour_b) > closest_index
        )

      neighbor_a_index = (index_a + direction_a) % len(contour_a)
      neighbor_b_index = (closest_index + direction_b) % len(contour_b)
      bridge_intersection, _, _ = segments_intersect(
        fixed_contour_a[index_a],
        fixed_contour_b[closest_index],
        fixed_contour_a[neighbor_a_index],
        fixed_contour_b[neighbor_b_index]
      )

      if bridge_intersection:
        continue

    if neighbor_a_is_positive and neighbor_b_is_positive:
      contours[self.contour_id1] = np.concatenate(
        (
          contours[self.contour_id1][:index_a + 1],
          contours[self.contour_id2][closest_index::-1],
          contours[self.contour_id2][:neighbor_b_index + 1:-1],
          contours[self.contour_id1][neighbor_a_index:],
        )
      )
      contours[self.contour_id2] = np.array([], np.int32)
    elif neighbor_a_is_positive and not neighbor_b_is_positive:
      pass
    elif not neighbor_a_is_positive and neighbor_b_is_positive:
      pass
    elif not neighbor_a_is_positive and not neighbor_b_is_positive:
      pass

      contours[self.contour_id1] = np.concatenate(
        (
          contours[self.contour_id1][:index_a],
          contours[self.contour_id2][closest_index + 1:],
          contours[self.contour_id2][:closest_index + 1],
          contours[self.contour_id1][index_a - 1:],
        )
      )
      contours[self.contour_id2] = np.array([], np.int32)
    else:
      contours[self.contour_id1] = np.concatenate(
        (
          contours[self.contour_id1][:index_a],
          contours[self.contour_id2][:closest_index],
          contours[self.contour_id2][:closest_index - 1:-1],
          contours[self.contour_id1][index_a:],
        ),
        axis=0
      )
      contours[self.contour_id2] = np.array([], np.int32)

    return contours
