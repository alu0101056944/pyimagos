'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of match point for each measurement of each expected contour.
Using fourier transform instead of hu_moments
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from src.main_execute import (
  create_minimal_image_from_contours
)
from src.expected_contours.expected_contour_of_branch import (
  ExpectedContourOfBranch
)
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.proximal_phalanx import ExpectedContourProximalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.sesamoid import ExpectedContourSesamoid
from src.expected_contours.metacarpal_sesamoid import (
  ExpectedContourSesamoidMetacarpal
)

def extract_sub_contours(contour, step=10):
  sub_contours = []

  for i in range(len(contour)):
    start = i
    end = start + step
    
    if end > len(contour):
      remaining = end - len(contour)
      sub_contour = np.vstack((contour[start:], contour[:remaining]))
    else:
      sub_contour = contour[start:end]
    
    sub_contours.append(sub_contour)

  return sub_contours

def resample_contour_fixed(contour, image_width, image_height, num_points=64):
  contour = contour.reshape((-1, 2))
  if len(contour) < 2:
    return contour

  distances = np.cumsum(
    np.sqrt(np.sum(np.diff(contour, axis=0, prepend=contour[-1:]) ** 2, axis=1))
  )
  total_distance = distances[-1]

  if total_distance == 0:
    return contour
  
  with np.errstate(invalid='ignore'): # floating point errors will result in NaN
    start_fill = contour[0, 0]
    end_fill = contour[-1, 0]
    fx = interp1d(
      distances,
      contour[:, 0],
      kind='linear',
      bounds_error=False, # Assign fill to out of bounds
      fill_value=(start_fill, end_fill),
    )
    fy = interp1d(
      distances,
      contour[:, 1],
      kind='linear',
      bounds_error=False,
      fill_value=(contour[0, 1], contour[-1, 1]),
    )

  new_distances = np.linspace(0, total_distance, num_points)
  x_new = fx(new_distances)
  y_new = fy(new_distances)

  # Only keep valid values
  valid_mask = ~(np.isnan(x_new) | np.isnan(y_new))
  x_new = x_new[valid_mask]
  y_new = y_new[valid_mask]

  # Clamp coordinates to be within image size
  x_new = np.clip(x_new, 0, image_width - 1)
  y_new = np.clip(y_new, 0, image_height - 1)

  resampled = np.column_stack((
      np.round(x_new).astype(int),
      np.round(y_new).astype(int)
  ))

  actual = resampled[:-1]
  next = resampled[1:]
  comparison_mask = actual == next
  fix_mixed_comparison_mask = ~np.all(comparison_mask, axis=1)
  prepended_first_always_true = np.r_[True, fix_mixed_comparison_mask]
  uniques_mask = prepended_first_always_true

  return resampled[uniques_mask]

def calculate_fourier_descriptors(contour, num_descriptors=10,
                                  defect_weight=0.3):
  contour = contour.astype(np.float32)
  complex_contour = contour[:, 0] + 1j * contour[:, 1]
  
  # Calculate Discrete Fourier Transform
  fourier = np.fft.fft(complex_contour)
  
  # fourier[0] = 0  # Remove translation information
  # magnitudes = np.abs(fourier)

  normalized = fourier / (np.abs(fourier) + 1e-9)

  # Prioritize higher descriptors, which focus more on finer details
  weights = np.linspace(0.5, 1.5, len(normalized))
  weighted_fourier = normalized * weights

  contour = contour.astype(np.int32)
  hull = cv.convexHull(contour, returnPoints=False)
  hull[::-1].sort(axis=0)
  defects = cv.convexityDefects(contour, hull) if len(contour) > 3 else None
  
  defect_score = 0
  if defects is not None:
    defect_depths = defects[:,0][:,3] / 256.0  # Normalized depth
    defect_score = np.mean(defect_depths) * defect_weight
  
  descriptors = np.concatenate([
      np.abs(weighted_fourier[1:num_descriptors + 1]), 
      [defect_score]
  ])

  return descriptors

def get_measurement_contour_match(contour, target_shape,
                                  image_width, image_height):
  resampled_target_shape = resample_contour_fixed(
    target_shape,
    image_width,
    image_height,
    128,
  )

  target_shape_descriptor = calculate_fourier_descriptors(
    resampled_target_shape,
    25,
  )

  best_match_start_index = -1
  best_match = None
  best_similarity = float('inf')
  subcontours = extract_sub_contours(contour, step=len(target_shape))
  for i, subcontour in enumerate(subcontours):
    resampled_subcontour = resample_contour_fixed(
      subcontour,
      image_width,
      image_height,
      128,
    )

    subcontour_descriptor = calculate_fourier_descriptors(
      resampled_subcontour,
      25,
    )
    defect_diff = np.abs(
      subcontour_descriptor[1] - target_shape_descriptor[1]
    )
    fourier_diff = np.linalg.norm(
      subcontour_descriptor[0] - target_shape_descriptor[0]
    )

    similarity = 0.7 * fourier_diff + 0.3 * defect_diff
    if similarity < best_similarity:
      best_similarity = similarity
      best_match_start_index = i
      best_match = subcontour

  return best_match, best_match_start_index

def show_contours_measurement_match(
    original_contour: np.array,
    target_shape: np.array,
    padding=5,
    title='Position restrictions visualization',
):
  original_contour = np.reshape(original_contour, (-1, 2))
  x_values = original_contour[:, 0]
  y_values = original_contour[:, 1]

  min_x = int(np.min(x_values))
  max_x = int(np.max(x_values))
  min_y = int(np.min(y_values))
  max_y = int(np.max(y_values))
  original_contour = original_contour - np.array([min_x, min_y])

  blank_image = np.zeros((max_y + 300, max_x + 25, 3), dtype=np.uint8)

  # Image: original contour
  (
    x1,
    y1,
    x2,
    y2
  ) = create_minimal_image_from_contours([original_contour])
  minimum_image = blank_image[y1:y2, x1:x2]
  minimum_image = cv.copyMakeBorder(
    minimum_image,
    padding,
    padding,
    padding,
    padding,
    cv.BORDER_CONSTANT,
    value=(0, 0, 0)
  )

  for i, contour in enumerate([original_contour]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(minimum_image, [original_contour], i, color, 1)

  # Image: target shape
  target_shape = np.reshape(target_shape, (-1, 2))
  x_values = target_shape[:, 0]
  y_values = target_shape[:, 1]

  max_x = int(np.max(x_values))
  min_x = int(np.min(x_values))
  max_y = int(np.max(y_values))
  min_y = int(np.min(y_values))
  width = max_x - min_x
  height = max_y - min_y
  target_shape = target_shape - np.array([min_x, min_y])

  height_difference = abs(minimum_image.shape[0] - height)
  corrected_height = min(minimum_image.shape[0], height + height_difference)
  target_shape_image = np.zeros((corrected_height, width + padding, 3),
                                dtype=np.uint8)
  for i, contour in enumerate([target_shape]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(target_shape_image, [target_shape], i, color, 1)

  # Image: only points in target shape
  target_shape_points_image = np.zeros(
    (minimum_image.shape[0], width, 3),
    dtype=np.uint8
  )
  color = (255, 0, 0)
  for point in target_shape:
    if len(contour) > 0:
      cv.circle(target_shape_points_image, point, 1, color, -1)

  # Image: resampled target shape to 64 points
  resampled_target_shape_points_image = np.zeros(
    (minimum_image.shape[0], width, 3),
    dtype=np.uint8
  )
  color = (255, 0, 0)
  resampled_target_shape = resample_contour_fixed(
    target_shape,
    image_width=width,
    image_height=minimum_image.shape[0],
    num_points=128,
  )
  for i, contour in enumerate([resampled_target_shape]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(
        resampled_target_shape_points_image,
        [resampled_target_shape],
        i,
        color,
        1
      )

  # Image: roi of original contour that matches
  match_contour, _ = get_measurement_contour_match(
    original_contour,
    target_shape,
    minimum_image.shape[1],
    minimum_image.shape[0],
  )
  match_image = np.zeros((minimum_image.shape[0], minimum_image.shape[1], 3),
                         dtype=np.uint8)
  for i, contour in enumerate([match_contour]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(match_image, [match_contour], i, color, 1)

  # Image: points of original contour
  original_contour_points_image = np.zeros(
    (minimum_image.shape[0], minimum_image.shape[1], 3),
    dtype=np.uint8
  )
  for point in original_contour:
    if len(contour) > 0:
      cv.circle(original_contour_points_image, point, 1, (255, 0, 0), -1)

  # Image: Measurement point on top of original shape
  measurement_point = match_contour[len(match_contour // 2) - 1]
  measurement_point_overlap_image = np.zeros(
    (minimum_image.shape[0], minimum_image.shape[1], 3),
    dtype=np.uint8
  )
  for i, contour in enumerate([original_contour]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(measurement_point_overlap_image, [original_contour], i, color, 1)

  cv.circle(measurement_point_overlap_image, measurement_point, 1,
            (0, 255, 255), -1)

  # Image: all subcontours
  subcontours = extract_sub_contours(original_contour, step=len(target_shape))
  subcontours_image = np.zeros(
    (minimum_image.shape[0], minimum_image.shape[1], 3),
    dtype=np.uint8
  )
  for i in range(1, len(subcontours), 6):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(subcontours_image, subcontours, i, color, 1)

  separator_color = (255, 255, 255)
  SEPARATOR_WIDTH = 2
  separator_column = np.full(
    (minimum_image.shape[0], SEPARATOR_WIDTH, 3),
    separator_color,
    dtype=np.uint8
  )

  concatenated = np.concatenate(
    (
      target_shape_image,
      separator_column,
      target_shape_points_image,
      separator_column,
      resampled_target_shape_points_image,
      separator_column,
      minimum_image,
      separator_column,
      original_contour_points_image,
      separator_column,
      match_image,
      separator_column,
      measurement_point_overlap_image,
      separator_column,
      subcontours_image,
    ),
    axis=1
  )

  fig = plt.figure()
  plt.imshow(concatenated)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def visualize_shape_match_fourier():
  metacarpal1 = np.array(
    [[[ 73, 211]],
    [[ 72, 212]],
    [[ 68, 212]],
    [[ 67, 213]],
    [[ 65, 213]],
    [[ 64, 214]],
    [[ 63, 214]],
    [[ 62, 215]],
    [[ 61, 215]],
    [[ 59, 217]],
    [[ 59, 218]],
    [[ 58, 219]],
    [[ 58, 220]],
    [[ 59, 221]],
    [[ 59, 225]],
    [[ 58, 226]],
    [[ 58, 229]],
    [[ 62, 233]],
    [[ 62, 234]],
    [[ 65, 237]],
    [[ 65, 238]],
    [[ 67, 240]],
    [[ 67, 241]],
    [[ 70, 244]],
    [[ 70, 245]],
    [[ 72, 247]],
    [[ 72, 248]],
    [[ 73, 249]],
    [[ 73, 250]],
    [[ 74, 251]],
    [[ 74, 252]],
    [[ 75, 253]],
    [[ 75, 254]],
    [[ 77, 256]],
    [[ 77, 257]],
    [[ 78, 258]],
    [[ 78, 259]],
    [[ 79, 260]],
    [[ 79, 261]],
    [[ 80, 262]],
    [[ 80, 263]],
    [[ 81, 264]],
    [[ 81, 265]],
    [[ 82, 266]],
    [[ 82, 267]],
    [[ 83, 268]],
    [[ 83, 269]],
    [[ 84, 270]],
    [[ 84, 271]],
    [[ 85, 272]],
    [[ 85, 273]],
    [[ 86, 274]],
    [[ 86, 276]],
    [[ 87, 277]],
    [[ 87, 280]],
    [[ 88, 281]],
    [[ 88, 285]],
    [[ 89, 286]],
    [[ 89, 295]],
    [[ 90, 296]],
    [[ 90, 298]],
    [[ 91, 299]],
    [[ 91, 300]],
    [[ 94, 303]],
    [[ 96, 303]],
    [[ 97, 304]],
    [[100, 304]],
    [[101, 305]],
    [[103, 305]],
    [[104, 304]],
    [[105, 304]],
    [[106, 303]],
    [[106, 302]],
    [[108, 300]],
    [[108, 299]],
    [[110, 297]],
    [[110, 296]],
    [[113, 293]],
    [[113, 292]],
    [[115, 290]],
    [[115, 289]],
    [[117, 287]],
    [[117, 286]],
    [[118, 285]],
    [[116, 283]],
    [[116, 282]],
    [[115, 282]],
    [[106, 273]],
    [[106, 272]],
    [[104, 270]],
    [[104, 269]],
    [[102, 267]],
    [[102, 266]],
    [[100, 264]],
    [[100, 263]],
    [[ 99, 262]],
    [[ 99, 261]],
    [[ 98, 260]],
    [[ 98, 259]],
    [[ 97, 258]],
    [[ 97, 257]],
    [[ 96, 256]],
    [[ 96, 255]],
    [[ 95, 254]],
    [[ 95, 253]],
    [[ 94, 252]],
    [[ 94, 250]],
    [[ 93, 249]],
    [[ 93, 248]],
    [[ 92, 247]],
    [[ 92, 246]],
    [[ 91, 245]],
    [[ 91, 243]],
    [[ 90, 242]],
    [[ 90, 240]],
    [[ 89, 239]],
    [[ 89, 237]],
    [[ 88, 236]],
    [[ 88, 227]],
    [[ 87, 226]],
    [[ 87, 223]],
    [[ 86, 222]],
    [[ 86, 220]],
    [[ 78, 212]],
    [[ 76, 212]],
    [[ 75, 211]]],
    dtype=np.int32
  )
  target_shape = np.array(
    [[[ 63, 214]],
    [[ 62, 215]],
    [[ 61, 215]],
    [[ 59, 217]],
    [[ 59, 218]],
    [[ 58, 219]],
    [[ 58, 220]],
    [[ 59, 221]],
    [[ 59, 225]],
    [[ 58, 226]],
    [[ 58, 229]],
    [[ 62, 233]],
    [[ 62, 234]],
    [[ 63, 235]],
    [[ 64, 235]],
    [[ 65, 214]]],
    dtype=np.int32
  )
  show_contours_measurement_match(
    metacarpal1,
    target_shape,
    padding=5,
    title='Composition metacarpal1 only.',
  )
  plt.show()
