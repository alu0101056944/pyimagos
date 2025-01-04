'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Find closest point from contour a to contour b and invade contour b from
the first point and one neighbour of it (which is +i or -i or = depending on the
id of the point within contour a) by taking <invasion_count> amount of neighbor
points in the contour b besides the closest point and the opposite to it.

The result can be visualized as a straight "bridge" that penetrates contour b
up to the opposite side and extends <invasion_count> laterally.
'''

import numpy as np

from src.contour_operations.utils import find_opposite_point_with_normals
from src.contour_operations.contour_operation import ContourOperation
from src.contour_operations.utils import segments_intersect

class ExtendContour(ContourOperation):
  def __init__(self, contour_id1: int, contour_id2: int, image_width: int,
               image_height: int, invasion_count: int):
    self.contour_id1 = contour_id1
    self.contour_id2 = contour_id2
    self.image_width = image_width
    self.image_height = image_height
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
    contours = list(contours)

    contour_a = contours[self.contour_id1]
    contour_b = contours[self.contour_id2]
    fixed_contour_a = np.reshape(contour_a, (-1, 2))
    fixed_contour_b = np.reshape(contour_b, (-1, 2))

    if len(contour_b) < 1:
      contours[self.contour_id2] = np.array([], dtype=np.int64)
      return contours

    index_a, closest_index, second_index = self._find_closest_pair(
      fixed_contour_a,
      fixed_contour_b
    )

    # duplicate index_a so that one is start and the other is end
    contour_a = np.insert(
      contour_a,
      index_a + 1,
      contour_a[index_a],
      axis=0
    )

    if len(contour_b) < 3:
      # Three points is just short of minimum 3 to be able to calculate the normal
      # because the normal needs the start point and next point and previous
      # point and because those are exactly three points then the normal will
      # not intersect

      contour_b = np.insert(
        contour_b,
        closest_index + 1,
        contour_b[closest_index],
        axis=0
      )
      
      contour_b_new = [
        *contour_b[closest_index::-1].tolist(),
        *contour_b[closest_index + 1:].tolist(),
      ],

      contour_a = np.insert(contour_a, index_a + 1, *contour_b_new, axis=0)
      
      contours[self.contour_id2] = np.array([], dtype=np.int64)
      contours[self.contour_id1] = contour_a
      return contours

    opposite_index_b = find_opposite_point_with_normals(
      fixed_contour_b,
      closest_index,
      self.image_width,
      self.image_height
    )

    # track 1 is opposite negative and closest positive
    # track 2 is opposite positive and closest negative
    # if they keep their direction and the invasion count is big enough they
    # will match or cross.
    track_1_opposite_negative = []
    track_1_closest_positive = []
    track_2_opposite_positive = []
    track_2_closest_negative = []
    match_at_track_1 = False
    cross_at_track_1 = False
    match_at_track_2 = False
    cross_at_track_2 = False
    for i in range(1, self.invasion_count + 1):
      if (match_at_track_1 or cross_at_track_1) and (
          match_at_track_2 or cross_at_track_2):
        break

      opposite_positive_next = fixed_contour_b[(opposite_index_b + i) % len(fixed_contour_b)]
      opposite_negative_next = fixed_contour_b[(opposite_index_b - i) % len(fixed_contour_b)]

      closest_positive_next = fixed_contour_b[(closest_index + i) % len(fixed_contour_b)]
      closest_negative_next = fixed_contour_b[(closest_index - i) % len(fixed_contour_b)]

      # When the result in the same track is the same point
      result_match_track_1 = (
        np.array_equal(opposite_negative_next, closest_positive_next)
      )
      result_match_track_2 = (
        np.array_equal(opposite_positive_next, closest_negative_next)
      )

      result_cross_track_1 = (
        (opposite_index_b - i) % len(fixed_contour_b) < (
          (closest_index + i) % len(fixed_contour_b)
        )
      )
      result_cross_track_2 = (
        (opposite_index_b + i) % len(fixed_contour_b) < (
          (closest_index - i) % len(fixed_contour_b)
        )
      )

      if not match_at_track_1 and not cross_at_track_1:
        if result_match_track_1:
          track_1_closest_positive.append(
            (closest_index + i) % len(fixed_contour_b)
          )
          match_at_track_1 = True
        elif result_cross_track_1:
          cross_at_track_1 = True
        else:
          track_1_opposite_negative.insert(
            0,
            (opposite_index_b - i) % len(fixed_contour_b)
          )
          track_1_closest_positive.append(
            (closest_index + i) % len(fixed_contour_b)
          )

      if not match_at_track_2 and not cross_at_track_2:
        if result_match_track_2:
          track_2_opposite_positive.append(
            (closest_index - i) % len(fixed_contour_b)
          )
          match_at_track_2 = True
        elif result_cross_track_2:
          cross_at_track_2 = True
        else:
          track_2_closest_negative.insert(
            0,
            (closest_index - i) % len(fixed_contour_b)
          )
          track_2_opposite_positive.append(
            (opposite_index_b + i) % len(fixed_contour_b)
          )

    contour_b_partial_indices = [
      closest_index,
      *track_1_closest_positive,
      *track_1_opposite_negative,
      opposite_index_b,
      *track_2_opposite_positive,
      *track_2_closest_negative,
    ]

    contour_b_partial = np.array(
      [np.copy(contour_b[i]) for i in contour_b_partial_indices],
      dtype=np.int64
    )

    # Insert duplicated closest_point so that bridge between contour a and
    # contour b is from index_a to closest_point in parallel.
    contour_b_partial = np.insert(
      contour_b_partial,
      len(contour_b_partial),
      contour_b_partial[0],
      axis=0
    )

    contour_a = np.insert(
      contour_a, index_a + 1, contour_b_partial,
      axis=0
    )

    # Contour b will have the invaded points deleted
    contour_b = np.delete(contour_b, contour_b_partial_indices, axis=0)

    contours[self.contour_id1] = contour_a
    contours[self.contour_id2] = contour_b

    return contours
