'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Contour operation related utils.
'''

import numpy as np
from scipy.spatial import distance
import cv2 as cv

def line_segment_intersection(line_start, line_end, segment_start,
                                segment_end, epsilon=1e-8):
  line_direction = line_end - line_start
  segment_direction = segment_end - segment_start
  det = -line_direction[0] * segment_direction[1] + (
    line_direction[1] * segment_direction[0]
  )

  if(abs(det) <= epsilon):
    return None, None # Lines are parallel

  # Solve matrix [t, u]
  matrix = np.array([[segment_direction[0], -line_direction[0]],
                    [segment_direction[1], -line_direction[1]]])
  inverse_matrix = np.linalg.inv(matrix)
  b = line_start - segment_start
  [u, t] = np.dot(inverse_matrix, b)

  if(t >= -epsilon and t <= 1 + epsilon and u >= -epsilon and u <= 1 + epsilon):
    return u, t # Intersection found
  return None, None

def find_intersection_points(line_start, direction, contour,
                             projection_distance) -> list:
  num_points = len(contour)
  intersections = []
  
  line_end_1 = line_start + direction * projection_distance
  line_end_2 = line_start - direction * projection_distance

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
        if not np.all(intersection_point.astype(np.int32) == line_start):
          intersections.append(intersection_point)
    
    if intersection_2[0] != None and intersection_2[1] != None:
        u, t = intersection_2
        intersection_point = segment_start + u * (segment_end - segment_start)
        if not np.all(intersection_point.astype(np.int32) == line_start):
          intersections.append(intersection_point)
  
  return intersections

def find_intersections_by_centroid(contour, start_point,
                                   projection_distance) -> list:
  global_centroid = np.mean(contour, axis=0)
  direction = global_centroid - start_point
  if np.linalg.norm(direction) == 0:
    return None
  direction = direction / np.linalg.norm(direction)

  intersections = find_intersection_points(start_point, direction, contour,
                                           projection_distance)
  return intersections

def find_intersections_by_neighbors(contour, start_point_index,
                                    projection_distance,
                                    num_neighbours=4) -> list:
  num_points = len(contour)
  start_point = contour[start_point_index]

  neighbor_vectors = []
  for i in range(num_neighbours):
    neighbor_a_index = (start_point_index + i) % num_points
    neighbor_b_index = (start_point_index - i) % num_points
    
    neighbor_a = contour[neighbor_a_index]
    neighbor_b = contour[neighbor_b_index]

    vector_a = neighbor_a - start_point
    vector_b = neighbor_b - start_point
    
    neighbor_vectors.append(vector_a)
    neighbor_vectors.append(vector_b)

  if not neighbor_vectors:
    return None

  local_centroid = np.mean(neighbor_vectors, axis=0)
  if np.linalg.norm(local_centroid) == 0:
    return None
  direction = local_centroid / np.linalg.norm(local_centroid)

  intersections = find_intersection_points(start_point, direction, contour,
                                           projection_distance)
  return intersections

def filter_internal_interceptions(intersections, start_point, contour) -> list:
  if intersections is None:
    return None
  internal_interceptions = []
  for intersection_point in intersections:
    p1 = np.array(start_point, dtype=np.float32) # float32 for no precision error
    p2 = np.array(intersection_point, dtype=np.float32)

    num_samples = 8
    sampled_points = []
    for i in range(num_samples + 1):
      alpha = i / num_samples
      x = int((1 - alpha) * p1[0] + alpha * p2[0])
      y = int((1 - alpha) * p1[1] + alpha * p2[1])
      sampled_points.append((x, y))

    inside_count = 0
    for point in sampled_points:
      result = cv.pointPolygonTest(contour.astype(np.float32),
                                   np.array(point, dtype=np.float32).tolist(),
                                   False)
      if result >= 0:  # Consider points on the edge as inside
        inside_count += 1

    if inside_count > num_samples:
      internal_interceptions.append(intersection_point)
  
  return internal_interceptions

def find_opposite_point(contour, start_point_idx, image_width, image_height):
  if len(contour) < 3:
    return None
  
  contour = np.reshape(contour, (-1, 2), copy=True)
  
  if len(np.unique(contour, axis=0)) < 3:
    return None

  start_point = contour[start_point_idx]

  maximum_distance = ((image_width ** 2) + (image_height ** 2)) ** 0.5
  projection_distance = maximum_distance

  intersections = find_intersections_by_centroid(
    contour,
    start_point,
    projection_distance
  )
  internal_interceptions = filter_internal_interceptions(
    intersections,
    start_point,
    contour
  )

  if not internal_interceptions:
    num_neighbors = [4, 2, 3]
    for neighbor_amount in num_neighbors:
      intersections = find_intersections_by_neighbors(
        contour,
        start_point_idx,
        projection_distance,
        neighbor_amount
      )
      internal_interceptions = filter_internal_interceptions(
        intersections,
        start_point,
        contour
      )
      if internal_interceptions:
        break

  # Select the opposite point
  if internal_interceptions:
    distances_to_start = [distance.euclidean(start_point, intersection)
        for intersection in internal_interceptions]
    intersection_point_idx = np.argmin(distances_to_start)
    intersection_point = internal_interceptions[intersection_point_idx]
    
    # Search the closest idx in the original contour to that intersection point
    dists = distance.cdist([intersection_point], contour)
    opposite_point_idx = np.argmin(dists)
  else:
    opposite_point_idx = None

  return opposite_point_idx

def segments_intersect(seg1_p1, seg1_p2, seg2_p1, seg2_p2, epsilon=1e-8):
  """Checks if two line segments intersect."""
  x1, y1 = seg1_p1
  x2, y2 = seg1_p2
  x3, y3 = seg2_p1
  x4, y4 = seg2_p2

  denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)

  if abs(denom) <= epsilon: # Parallel
      return False, None, None

  t_num = (x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)
  u_num = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)

  t = t_num / denom
  u = u_num / denom
  return (t > -epsilon and
          t <= 1 + epsilon and
          u > -epsilon and
          u <= 1 + epsilon), t, u

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
