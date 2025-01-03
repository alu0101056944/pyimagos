'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Cut a given contour id by a specific point by calculating an opposite point
using the normal and the nearest point of intersection point to start point
and then selecting the subsequence from start point to opposite point,
leaving the rest as another contour.
'''

import numpy as np
from src.contour_operations.utils import find_opposite_point_with_normals

from src.contour_operations.contour_operation import ContourOperation

class CutContour(ContourOperation):
  def __init__(self, contour_id: int, cut_point_id: int, image_width: int,
               image_height):
    self.contour_id = contour_id
    self.cut_point_id = cut_point_id
    self.image_width = image_width
    self.image_height = image_height
    
  def _split_contour_by_indices(self, contour, start_point_idx,
                                opposite_point_idx):
    split_indices = sorted([start_point_idx, opposite_point_idx])
    
    contour_1 = np.concatenate((contour[0:split_indices[0] + 1],
                                contour[split_indices[1]:]))
    contour_2 = contour[split_indices[0]:split_indices[1] + 1]
    return contour_1, contour_2

  def generate_new_contour(self, contours: list) -> list:
    contours = list(contours)

    opposite_point_idx = find_opposite_point_with_normals(
      contours[self.contour_id],
      self.cut_point_id,
      self.image_width,
      self.image_height
    )

    contour_1, contour_2 = self._split_contour_by_indices(
      contours[self.contour_id],
      self.cut_point_id,
      opposite_point_idx
    )

    contours[self.contour_id] = contour_1
    contours.insert(self.contour_id + 1, contour_2)
    return contours
