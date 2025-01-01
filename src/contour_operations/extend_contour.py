'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development


'''

import numpy as np

from src.contour_operations.utils import find_opposite_point_with_normals
from src.contour_operations.contour_operation import ContourOperation

class ExtendContour(ContourOperation):
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

    if len(contour_b) < 3:
      # Three points is just short of minimum 4 to be able to calculate the normal
      # because the normal needs the start point and next point and previous
      # point and because those are exactly three points then the normal will
      # not intersect
      return np.insert(contour_a, index_a + 1, contour_b)

    opposite_index_b = find_opposite_point_with_normals(
      fixed_contour_a,
      closest_index,
      self.projection_distance
    )

    if not opposite_index_b:
      # probably a rect contour, very rare, because if there are more than three
      # points and no opposite is found then all points are parallel
      return np.insert(contour_a, index_a + 1, contour_b)

    # From the pov of the closest_b being the center, it's opposite
    # being the center on the opposite side. The idea is to
    # concatenate the different subsequences (tracks) and insert it
    # into the contour a, so that the closest center has at one side
    # the - i track, at the other the + i track, same for opposite.
    opposite_point_track_a = [opposite_index_b]
    opposite_point_track_b = []
    closest_point_track_a = [closest_index]
    closest_point_track_b = []
    matched_at_a = False
    looped_at_a = False
    matched_at_b = False
    looped_at_b = False
    for i in range(self.invasion_count):
      opposite_direction_result_a = (
        contour_b[(opposite_index_b + i) % len(contour_b)]
      )
      opposite_direction_result_b = (
        contour_b[(opposite_index_b - i) % len(contour_b)]
      )

      closest_direction_result_a = (
        contour_b[(closest_index + i) % len(contour_b)]
      )
      closest_direction_result_b = (
        contour_b[(closest_index - i) % len(contour_b)]
      )

      # When the -i or +i will end up in the same point (so only one point
      # inbetween)
      direction_result_match_a = (
        opposite_direction_result_a == (
          closest_direction_result_a or closest_direction_result_b
        )
      )
      direction_result_match_b = (
        opposite_direction_result_b == (
          closest_direction_result_a or closest_direction_result_b
        )
      )

      # When the -i or +i reach the respective opposite (closest reaches
      # opposite and viceversa), so there are no points inbetween them.
      direction_result_loop_a = (
        opposite_direction_result_a == contour_b[closest_index]
      )
      direction_result_loop_b = (
        opposite_direction_result_b == contour_b[closest_index]
      )

      if not matched_at_a and not looped_at_a:
        if direction_result_match_a:
          closest_point_track_a.insert(0, (closest_index + i) % len(contour_b))
          matched_at_a = True
        elif direction_result_loop_a:
          looped_at_a = True
        else:
          closest_point_track_a.insert(0, (closest_index + i) % len(contour_b))
          opposite_point_track_a.insert(0, (opposite_index_b + i) % len(contour_b))
        
      if not matched_at_b and not looped_at_b:
        if direction_result_match_b:
          closest_point_track_b.append((closest_index - i) % len(contour_b))
          matched_at_b = True
        elif direction_result_loop_b:
          looped_at_b = True
        else:
          closest_point_track_b.append((closest_index - i) % len(contour_b))
          opposite_point_track_b.append((opposite_index_b - i) % len(contour_b))

    contour_b_partial_indices = np.concatenate(
      (
        closest_point_track_b,
        opposite_point_track_b,
        opposite_point_track_a,
        closest_point_track_a
      )
    )
    contour_b_partial = [contour_b[i] for i in contour_b_partial_indices]
    contours[self.contour_id1] = np.insert(
      contours[self.contour_id1], closest_index + 1, contour_b_partial
    )

    contour_b = np.delete(contour_b, contour_b_partial_indices, axis=0)

    return contours
