'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Contour operation related utils.
'''

import numpy as np
from scipy.spatial import distance

def line_segment_intersection(line_start, line_end, segment_start,
                                segment_end):
  '''Finds the intersection point between a line and a segment using parameters.
  Returns:
  t, u parameters or None if no intersection
  '''
  line_direction = line_end - line_start
  segment_direction = segment_end - segment_start
  det = -line_direction[0] * segment_direction[1] + (
    line_direction[1] * segment_direction[0]
  )

  if(det == 0):
    return None # Lines are parallel

  # Solve matrix [t, u]
  matrix = np.array([[segment_direction[0], -line_direction[0]],
                    [segment_direction[1], -line_direction[1]]])
  inverse_matrix = np.linalg.inv(matrix)
  b = line_start - segment_start
  [u, t] = np.dot(inverse_matrix, b)

  if(t >= 0 and t <= 1 and u >= 0 and u <= 1):
    return u, t # Intersection found
  return None

def find_opposite_point_with_normals(self, contour, start_point_idx,
                                      normal_projection_distance = 100):
  '''Finds the 'opposite' point using normal vector and intersection checks.'''
  start_point = contour[start_point_idx]
  num_points = len(contour)

  # Calculate the normal vector at the starting point
  point_prev_idx = (start_point_idx - 1) % num_points
  point_next_idx = (start_point_idx + 1) % num_points
  point_prev = contour[point_prev_idx]
  point_next = contour[point_next_idx]
  tangent = point_next - point_prev
  normal = np.array([-tangent[1], tangent[0]]) # Rotate 90 degrees

  # Normalize it
  normal = normal / np.linalg.norm(normal)
  
  # Project the normal vector into a line
  line_start = start_point
  line_end_1 = start_point + normal * normal_projection_distance
  line_end_2 = start_point - normal * normal_projection_distance

  # Find intersections
  intersections = []
  for i in range(num_points):
    segment_start = contour[i]
    segment_end = contour[(i +1 ) % num_points]

    # Check for intersection with the projected line in both directions
    intersection_1 = self._line_segment_intersection(
        line_start,
        line_end_1,
        segment_start,
        segment_end
    )
    intersection_2 = self._line_segment_intersection(
        line_start,
        line_end_2,
        segment_start,
        segment_end
    )

    if intersection_1:
        t, u = intersection_1
        intersection_point = segment_start + u * (segment_end - segment_start)
        intersections.append(intersection_point)
    
    if intersection_2:
        t, u = intersection_2
        intersection_point = segment_start + u * (segment_end - segment_start)
        intersections.append(intersection_point)

  # Select the opposite point
  if intersections:
    distances_to_start = [distance.euclidean(start_point, intersection)
        for intersection in intersections]
    intersection_point_idx = np.argmax(distances_to_start)
    intersection_point = intersections[intersection_point_idx]
    
    # Search the closest idx in the original contour to that intersection point
    dists = distance.cdist([intersection_point], contour)
    opposite_point_idx = np.argmin(dists)
    
  return opposite_point_idx
