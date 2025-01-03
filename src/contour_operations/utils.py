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
  line_direction = line_end - line_start
  segment_direction = segment_end - segment_start
  det = -line_direction[0] * segment_direction[1] + (
    line_direction[1] * segment_direction[0]
  )

  if(det == 0):
    return None, None # Lines are parallel

  # Solve matrix [t, u]
  matrix = np.array([[segment_direction[0], -line_direction[0]],
                    [segment_direction[1], -line_direction[1]]])
  inverse_matrix = np.linalg.inv(matrix)
  b = line_start - segment_start
  [u, t] = np.dot(inverse_matrix, b)

  if(t >= 0 and t <= 1 and u >= 0 and u <= 1):
    return u, t # Intersection found
  return None, None

def find_opposite_point_with_normals(contour, start_point_idx,
                                      image_width, image_height):
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
  
  maximum_distance = ((image_width ** 2) + (image_height ** 2)) ** 0.5
  normal_projection_distance = maximum_distance

  # Project the normal vector into a line
  line_start = start_point
  line_end_1 = start_point + normal * normal_projection_distance
  line_end_2 = start_point - normal * normal_projection_distance

  # Find intersections
  intersections = []
  for i in range(num_points):
    segment_start = contour[i]
    segment_end = contour[(i + 1) % num_points]

    # Check for intersection with the projected line in both directions
    intersection_1 = line_segment_intersection(
        line_start,
        line_end_1,
        segment_start,
        segment_end
    )
    intersection_2 = line_segment_intersection(
        line_start,
        line_end_2,
        segment_start,
        segment_end
    )

    if intersection_1[0] != None and intersection_1[1] != None:
        u, t = intersection_1
        intersection_point = segment_start + u * (segment_end - segment_start)
        if not np.all(intersection_point == line_start):
          intersections.append(intersection_point)
    
    if intersection_2[0] != None and intersection_2[1] != None:
        u, t = intersection_2
        intersection_point = segment_start + u * (segment_end - segment_start)
        if not np.all(intersection_point == line_start):
          intersections.append(intersection_point)

  # Select the opposite point
  if intersections:
    distances_to_start = [distance.euclidean(start_point, intersection)
        for intersection in intersections]
    intersection_point_idx = np.argmin(distances_to_start)
    intersection_point = intersections[intersection_point_idx]
    
    # Search the closest idx in the original contour to that intersection point
    dists = distance.cdist([intersection_point], contour)
    opposite_point_idx = np.argmin(dists)
    
  return opposite_point_idx

def segments_intersect(seg1_p1, seg1_p2, seg2_p1, seg2_p2):
  """Checks if two line segments intersect."""
  x1, y1 = seg1_p1
  x2, y2 = seg1_p2
  x3, y3 = seg2_p1
  x4, y4 = seg2_p2

  denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)

  if denom == 0: # Parallel
      return False, None, None

  t_num = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
  u_num = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

  t = t_num / denom
  u = u_num / denom
  return 0 <= t <= 1 and 0 <= u <= 1, t, u


def blend_colors_with_alpha(background_color: np.array,
                            foreground_color: np.array) -> tuple:
  background_color = background_color.astype(np.float32)
  foreground_color = foreground_color.astype(np.float32)

  alpha_new = foreground_color[3] / 255.0
  alpha_current = background_color[3] / 255.0

  blended_alpha = alpha_new + alpha_current * (1 - alpha_new)
  blended_color = (
      (
        alpha_new * foreground_color[:3] +
        (alpha_current * (1 - alpha_new) ) * background_color[:3] 
      ) / blended_alpha
      if blended_alpha != 0 else [0, 0, 0]
  )

  blended_color = blended_color.astype(np.uint8)
  blended_alpha = (blended_alpha * 255).astype(np.uint8)

  return blended_color, blended_alpha
