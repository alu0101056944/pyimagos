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
  def __init__(self, contour_id1: int, contour_id2: int,
               point_id1: int = None, point_id2: int = None):
    self.contour_id1 = contour_id1
    self.contour_id2 = contour_id2
    self.point_id1 = point_id1
    self.point_id2 = point_id2

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

    if len(contour_b) == 0:
      return None

    if len(contour_a) == 0:
      return None

    if self.point_id1 is not None:
      # calculate closest index in contour b to point1
      point1 = fixed_contour_a[self.point_id1]
      distances = np.sqrt(np.sum((fixed_contour_b - point1) ** 2, axis=1))
      sorted_indices = np.argsort(distances)

      index_a = self.point_id1
      closest_index = sorted_indices[0]
    elif self.point_id1 is not None and self.point_id2 is not None:
      index_a = self.point_id1
      closest_index = self.point_id2
    else:
      index_a, closest_index, _ = self._find_closest_pair(
        fixed_contour_a,
        fixed_contour_b
      )

    contour_a = np.insert(contour_a, index_a + 1, fixed_contour_a[index_a], axis=0)
    contour_b = np.insert(contour_b, closest_index + 1,
                          fixed_contour_b[closest_index], axis=0)

    contours[self.contour_id1] = np.concatenate(
      (
        contours[self.contour_id1][:index_a + 1],
        contours[self.contour_id2][closest_index:],
        contours[self.contour_id2][:closest_index + 1],
        contours[self.contour_id1][index_a:],
      )
    )
    contours[self.contour_id2] = np.array([], np.int32)
   
    return contours
