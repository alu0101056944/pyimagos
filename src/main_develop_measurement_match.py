'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of match point for each measurement of each expected contour
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

def get_measurement_contour_match(contour, target_shape): 
  best_match_start_index = -1
  best_match = None
  best_similarity = float('inf')
  subcontours = extract_sub_contours(contour, step=10)
  for i, subcontour in enumerate(subcontours):
    similarity = cv.matchShapes(subcontour, target_shape, cv.CONTOURS_MATCH_I1, 0)
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

  # Third image: original contour
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

  # First image: target shape
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

  # Second image: only points in target shape
  target_shape_points_image = np.zeros(
    (minimum_image.shape[0], minimum_image.shape[1], 3),
    dtype=np.uint8
  )
  color = (255, 0, 0)
  for point in target_shape:
    if len(contour) > 0:
      cv.circle(target_shape_points_image, point, 1, color, -1)

  # Fourth image: roi of original shape that matches
  match_contour, _ = get_measurement_contour_match(
    original_contour,
    target_shape,
  )
  match_image = np.zeros((minimum_image.shape[0], minimum_image.shape[1], 3),
                         dtype=np.uint8)
  for i, contour in enumerate([match_contour]):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(match_image, [match_contour], i, color, 1)

  # Fifth image: Measurement point on top of original shape
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

  # Sixth image: all subcontours
  subcontours = extract_sub_contours(original_contour, step=10)
  subcontours_image = np.zeros(
    (minimum_image.shape[0], minimum_image.shape[1], 3),
    dtype=np.uint8
  )
  for i, contour in enumerate(subcontours):
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
      minimum_image,
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

def visualize_shape_match():
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
  target_shape = np.array([0.47967672, 0.23311022, 0.10885446, 0.04182705,
                              0.0368656 , 0.03411718, 0.07823383, 0.08240334,
                              0.07883134, 0.0411524 ], dtype=np.float32)
  show_contours_measurement_match(
    metacarpal1,
    target_shape,
    padding=5,
    title='Composition metacarpal1 only.',
  )
  plt.show()
