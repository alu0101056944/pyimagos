'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for distal phalanx
'''

from enum import Enum, auto

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.main_develop_corner_order import get_top_left_corner

class BoundingBoxPoint(Enum):
  TOPLEFT = auto(),
  TOPRIGHT = auto(),
  BOTTOMRIGHT = auto(),
  BOTTOMLEFT = auto(),

def calculate_positional_image(
    bounding_rect_contour: list,
    min_rect,
    concatenated_image_width: int,
    image_height: int
) -> np.array:
  bounding_rect_contour = np.array(bounding_rect_contour) + 20
  image = np.full((image_height * 6, concatenated_image_width, 3), 0, dtype=np.uint8)

  cv.drawContours(image, [bounding_rect_contour], 0, (0, 255, 0), 1)

  top_left_corner, i = get_top_left_corner(
    bounding_rect_contour,
    concatenated_image_width,
    image_height
  )
  top_left_corner = top_left_corner
  top_right_corner = bounding_rect_contour[
    (i + 1) % len(bounding_rect_contour)
  ].tolist()
  bottom_right_corner = bounding_rect_contour[
    (i + 2) % len(bounding_rect_contour)
  ].tolist()
  bottom_left_corner = bounding_rect_contour[
    (i + 3) % len(bounding_rect_contour)
  ].tolist()

  ERROR_PADDING = 4
  point_1 = (bottom_right_corner + np.array([ERROR_PADDING, 0]))
  point_2 = (top_right_corner + np.array([ERROR_PADDING, 0]))
  direction = point_2 - point_1
  cv.line(
    image,
    point_1 - direction * concatenated_image_width,
    point_2 + direction * concatenated_image_width,
    (255, 255, 0),
    1
  )
  point_1 = (bottom_left_corner - np.array([ERROR_PADDING, 0]))
  point_2 = (top_left_corner - np.array([ERROR_PADDING, 0]))
  direction = point_2 - point_1
  cv.line(
    image,
    point_1 - direction * concatenated_image_width,
    point_2 + direction * concatenated_image_width,
    (255, 255, 0),
    1
  )
  point_1 = (bottom_left_corner + np.array([0, ERROR_PADDING]))
  point_2 = (bottom_right_corner + np.array([0, ERROR_PADDING]))
  direction = point_2 - point_1
  cv.line(
    image,
    point_1 - direction * concatenated_image_width,
    point_2 + direction * concatenated_image_width,
    (255, 255, 0),
    1
  )
  height = int(min_rect[1][1])
  bottom_bound = height * 4
  point_1 = np.array((bottom_left_corner + np.array([0, bottom_bound])))
  point_2 = np.array((bottom_right_corner + np.array([0, bottom_bound])))
  direction = point_2 - point_1
  cv.line(
    image,
    point_1 - direction * concatenated_image_width,
    point_2 + direction * concatenated_image_width,
    (255, 255, 0),
    1
  )

  cv.circle(image, top_left_corner, 1, (255, 0, 0), -1)
  cv.circle(image, top_right_corner, 1, (0, 0, 255), -1)
  cv.circle(image, bottom_right_corner, 1, (0, 255, 255), -1)
  cv.circle(image, bottom_left_corner, 1, (255, 0, 255), -1)

  return image

def create_minimal_image_from_contours(image: np.array,
                                       contours: list,
                                       padding = 0) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(max(0, np.min(x_values) - padding))
  min_y = int(max(0, np.min(y_values) - padding))
  max_x = int(min(image.shape[1], np.max(x_values) + padding))
  max_y = int(min(image.shape[0], np.max(y_values) + padding))

  roi_from_original = image[min_y:max_y + 1, min_x:max_x + 1]
  roi_from_original = np.copy(roi_from_original)

  # missing X padding correction on the left
  if np.min(x_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(x_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=1,
      dtype=np.uint8
    )
  
  # missing X padding correction on the right
  if np.max(x_values) + padding > image.shape[1]:
    missing_pixel_amount = np.absolute(np.max(x_values) + padding)
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8)
      ),
      axis=1,
      dtype=np.uint8
    )

  # missing Y padding correction on top
  if np.min(y_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(y_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=0,
      dtype=np.uint8
    )
  
  # missing Y padding correction on top
  if np.max(y_values) + padding > image.shape[0]:
    missing_pixel_amount = np.absolute(np.max(y_values) + padding)
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
      ),
      axis=0,
      dtype=np.uint8
    ) 

  corrected_contours = [
    points - np.array([[[min_x, min_y]]]) + padding for points in contours
  ]

  return roi_from_original, corrected_contours

def prepare_image_showing_shape(contours, approximated_contour, image_width,
                                image_height, title):

  # Separator (vertical)
  separator_color = (255, 255, 255)
  separator_width = 2
  separator_column = np.full(
    (image_height, separator_width, 3),
    separator_color,
    dtype=np.uint8
  )

  # First: blank image
  blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
  # Draw contours on blank image
  for i, contour in enumerate(contours):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(blank_image, contours, i, color, 1)

  # Second: Approximated image
  approximated_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
  cv.drawContours(approximated_image, [approximated_contour], 0, (200, 200, 0),
                  1)

  # Third: minimum bounding box
  bounding_box_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
  contour = np.reshape(contours[0], (-1, 2))
  rect = cv.minAreaRect(contour)
  bounding_rect_contour = cv.boxPoints(rect)
  bounding_rect_contour = np.int32(bounding_rect_contour) # to int
  cv.drawContours(bounding_box_image, [bounding_rect_contour], 0, (0, 255, 0),
                  1)
  
  # Fourth: convex hull
  convex_hull_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
  hull = cv.convexHull(contour)
  cv.drawContours(convex_hull_image, [hull], 0, (0, 255, 0), 1)

  # Fifth: hull convexity defects
  hull_area = cv.contourArea(hull)
  hull_defects_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
  cv.drawContours(hull_defects_image, [hull], 0, (0, 255, 0), 2)
  hull_indices = cv.convexHull(contour, returnPoints=False)
  defects = cv.convexityDefects(contour, hull_indices)
  if defects is not None:
    for i in range(defects.shape[0]):
      start_index, end_index, farthest_point_index, distance = defects[i, 0]

      start = contour[start_index]
      end = contour[end_index]
      farthest = contour[farthest_point_index]

      defect_area = cv.contourArea(np.array([start, end, farthest]))

      cv.line(hull_defects_image, start, end, (255, 0, 0), 1)
      if defect_area / hull_area > 0.1:
        cv.circle(hull_defects_image, farthest, 1, (0, 255, 255), -1)
      else:
        cv.circle(hull_defects_image, farthest, 1, (0, 140, 45), -1)

  concatenated = np.concatenate(
    (
      blank_image,
      separator_column,
      approximated_image,
      separator_column,
      bounding_box_image,
      separator_column,
      convex_hull_image,
      separator_column,
      hull_defects_image
    ),
    axis=1
  )

  positional_view_image = calculate_positional_image(bounding_rect_contour,
                                                     rect,
                                                     concatenated.shape[1],
                                                     image_height)

  # Separator (horizontal)
  separator_color = (255, 255, 255)
  separator_width = 2
  separator_column = np.full(
    (separator_width, concatenated.shape[1], 3),
    separator_color,
    dtype=np.uint8
  )

  concatenated = np.concatenate(
    (
      concatenated,
      positional_view_image
    ),
    axis=0
  )

  fig = plt.figure()
  plt.imshow(concatenated)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

  (
    area,
    aspect_ratio,
    solidity,
    significant_convex_hull_defects,
    hu_moments,
    difference
  ) = calculate_attributes(contour)
  text = f'area={area}\naspect_ratio={aspect_ratio}\n' \
    f'solidity={solidity}\n' \
    f'significant_convex_hull_defects={significant_convex_hull_defects}\n' \
    f'hu_moments={hu_moments}\n' \
    f'difference={difference}'
  plt.text(0, 1.23, text, transform=plt.gca().transAxes,
           verticalalignment='bottom', horizontalalignment='left')

def calculate_attributes(contour) -> list:
  contour = np.reshape(contour, (-1, 2))

  area = cv.contourArea(contour)

  rect = cv.minAreaRect(contour)
  width, height = rect[1]
  aspect_ratio = max(width, height) / min(width, height)

  solidity = (width * height) / (
      cv.contourArea(cv.convexHull(contour))
  )

  significant_convexity_defects = 0
  hull_area = cv.contourArea(cv.convexHull(contour))
  hull = cv.convexHull(contour, returnPoints=False)
  defects = cv.convexityDefects(contour, hull)
  if defects is not None:
    for i in range(defects.shape[0]):
      start_index, end_index, farthest_point_index, distance = defects[i, 0]

      start = contour[start_index]
      end = contour[end_index]
      farthest = contour[farthest_point_index]

      defect_area = cv.contourArea(np.array([start, end, farthest]))

      if defect_area / hull_area > 0.1:
        significant_convexity_defects += 1

  moments = cv.moments(contour)
  hu_moments = cv.HuMoments(moments)
  hu_moments = (np.log10(np.absolute(hu_moments))).flatten()

  reference_hu_moments = np.array(
    [
      -0.59893488,
      -1.62052591,
      -2.46926287,
      -3.46397177,
      -6.4447155,
      -4.28778216,
      -7.03097531
    ],
    dtype=np.float64
  )
  difference = np.linalg.norm(hu_moments - reference_hu_moments)

  return (
    area,
    aspect_ratio,
    solidity,
    significant_convexity_defects,
    hu_moments,
    difference
  )

def show_contour(contour, padding=0, title='Distal phalanx variation',
                 minimize_image: bool = True):
  contour = np.reshape(contour, (-1, 2))
  x_values = contour[:, 0]
  y_values = contour[:, 1]

  max_x = int(np.max(x_values))
  max_y = int(np.max(y_values))

  blank_image = np.zeros((max_y + 5, max_x + 5), dtype=np.uint8)

  if minimize_image:
    minimal_image, adjusted_contours = create_minimal_image_from_contours(
      blank_image,
      [contour],
      padding
    )
    minimal_image = cv.cvtColor(minimal_image, cv.COLOR_GRAY2RGB)
    contours = adjusted_contours
  else:
    minimal_image = blank_image
    contours = [contour]

  epsilon = 0.02 * cv.arcLength(contours[0], closed=True)
  approximated_contour = cv.approxPolyDP(contours[0], epsilon, True)
  approximated_contour = np.reshape(approximated_contour, (-1, 2))

  prepare_image_showing_shape(contours, approximated_contour,
                              image_width=minimal_image.shape[1],
                              image_height=minimal_image.shape[0],
                              title=title)

def visualize_distal_phalanx_shape():
  borders_detected = Image.open('docs/distal_phalanx.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], 5, 'Distal phalanx original.')
  
  under_80_area = np.array(
    [[[28,  80]],
    [[27,  80]],
    [[26,  80]],
    [[26,  80]],
    [[27,  82]],
    [[27,  83]],
    [[28,  84]],
    [[28,  85]],
    [[27,  86]],
    [[27,  88]],
    [[26,  89]],
    [[26,  90]],
    [[26,  91]],
    [[27,  92]],
    [[28,  92]],
    [[29,  92]],
    [[30,  92]],
    [[31,  92]],
    [[32,  92]],
    [[32,  92]],
    [[33,  91]],
    [[33,  91]],
    [[35,  91]],
    [[36,  90]],
    [[36,  89]],
    [[34,  88]],
    [[33,  86]],
    [[32,  85]],
    [[32,  84]],
    [[32,  83]],
    [[33,  82]],
    [[33,  80]],
    [[31,  80]],
    [[30,  79]],
    [[29,  80]]],
    dtype=np.int32
  )
  show_contour(under_80_area, 5, 'Under 80 area distal phalanx.')

  bad_aspect_ratio = np.array(
    [[[84, 120]],
    [[81, 120]],
    [[78, 120]],
    [[78, 120]],
    [[81, 123]],
    [[81, 123]],
    [[84, 126]],
    [[84, 126]],
    [[81, 129]],
    [[78, 132]],
    [[78, 137]],
    [[79, 138]],
    [[84, 138]],
    [[87, 138]],
    [[90, 138]],
    [[93, 138]],
    [[94, 138]],
    [[98, 135]],
    [[91, 126]],
    [[96, 123]],
    [[97, 120]],
    [[93, 120]],
    [[87, 120]]],
    dtype=np.int32
  )
  show_contour(bad_aspect_ratio, 5, 'Bad aspect ratio distal phalanx.')

  larger_aspect_contour = np.array([
    [[25, 59]],
    [[24, 60]],
    [[21, 60]],
    [[19, 62]],
    [[19, 65]],
    [[20, 66]],
    [[20, 67]],
    [[21, 68]],
    [[21, 73]],
    [[22, 74]],
    [[22, 81]],
    [[23, 82]],
    [[23, 83]],
    [[22, 84]],
    [[22, 87]],
    [[21, 88]],
    [[21, 89]],
    [[20, 90]],
    [[20, 91]],
    [[19, 92]],
    [[19, 96]],
    [[20, 97]],
    [[31, 97]],
    [[32, 96]],
    [[35, 96]],
    [[36, 95]],
    [[39, 95]],
    [[40, 94]],
    [[41, 94]],
    [[40, 93]],
    [[40, 92]],
    [[41, 91]],
    [[40, 90]],
    [[39, 90]],
    [[34, 85]],
    [[34, 84]],
    [[32, 82]],
    [[32, 79]],
    [[31, 78]],
    [[31, 68]],
    [[32, 67]],
    [[32, 61]],
    [[31, 60]],
    [[28, 60]],
    [[27, 59]]],
    dtype=np.int32
  )
  show_contour(larger_aspect_contour, 0,
               'Second occurrence non within range distal phalanx aspect' \
               ' ratio\'s.')

  high_solidity = np.array(
    [[[ 2,  0]],
    [[ 0,  2]],
    [[ 0,  6]],
    [[ 5, 11]],
    [[ 0, 16]],
    [[ 0, 26]],
    [[ 3, 29]],
    [[13, 29]],
    [[18, 24]],
    [[16, 22]],
    [[16, 21]],
    [[11, 16]],
    [[11, 15]],
    [[ 8, 12]],
    [[ 8,  0]]],
    dtype=np.int32
  )
  show_contour(high_solidity, 5, 'High solidity distal phalanx.')
  
  over_convex_defects = np.array(
    [[[ 2,  0]],
    [[ 0,  2]],
    [[ 0,  6]],
    [[ 5, 11]],
    [[ 0, 16]],
    [[ 0, 26]],
    [[ 3, 29]],
    [[13, 29]],
    [[18, 27]],
    [[18, 25]],
    [[8, 20]],
    [[18, 16]],
    [[18, 15]],
    [[9,  8]],
    [[9,  7]],
    [[14,  4]],
    [[14,  0]]],
    dtype=np.int32
  )
  show_contour(over_convex_defects, 5, 'Too many significant convexity defects ' \
               'distal phalanx.')

  under_convex_defects = np.array(
    [[[ 2,  0]],
    [[ 0,  2]],
    [[ 0,  6]],
    [[ 5, 11]],
    [[ 0, 16]],
    [[ 0, 26]],
    [[ 3, 29]],
    [[13, 29]],
    [[18, 24]],
    [[18,  1]],
    [[17,  0]]],
    dtype=np.int32
  )
  show_contour(under_convex_defects, 5, 'Too few significant convexity defects ' \
               'distal phalanx.')

  plt.show()
