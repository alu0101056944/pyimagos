'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for medial phalanx
'''

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.main_develop_test_distal_phalanx import (
  create_minimal_image_from_contours,
  calculate_positional_image,
)

def calculate_attributes(contour, show_convex_defects: bool = True) -> list:
  contour = np.reshape(contour, (-1, 2))

  area = cv.contourArea(contour)

  rect = cv.minAreaRect(contour)
  width, height = rect[1]
  aspect_ratio = max(width, height) / min(width, height)

  solidity = (width * height) / (
      cv.contourArea(cv.convexHull(contour))
  )

  if show_convex_defects:
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

        if defect_area / hull_area > 0.07:
          significant_convexity_defects += 1
  else:
    significant_convexity_defects = -1

  moments = cv.moments(contour)
  hu_moments = cv.HuMoments(moments)
  hu_moments = (np.log10(np.absolute(hu_moments))).flatten()

  reference_hu_moments = np.array(
    [
      -0.70441841,
      -2.11182333,
      -3.80250783,
      -4.94659795,
      -9.39066072,
      -6.30047546,
      -9.60233336
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

def prepare_image_showing_shape(contours, approximated_contour, image_width,
                                image_height, title, test_contour=None,
                                show_convex_defects: bool = True):

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
  if show_convex_defects:
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
        if defect_area / hull_area > 0.06:
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
  else:
    concatenated = np.concatenate(
      (
        blank_image,
        separator_column,
        approximated_image,
        separator_column,
        bounding_box_image,
        separator_column,
        convex_hull_image,
      ),
      axis=1
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
  ) = calculate_attributes(contour, show_convex_defects)
  text = f'area={area}\naspect_ratio={aspect_ratio}\n' \
    f'solidity={solidity}\n' \
    f'significant_convex_hull_defects={significant_convex_hull_defects}\n' \
    f'hu_moments={hu_moments}\n' \
    f'difference={difference}'
  plt.text(0, 1.24, text, transform=plt.gca().transAxes,
           verticalalignment='bottom', horizontalalignment='left')
  
  positional_view_image = calculate_positional_image(
                                                    contour,
                                                    bounding_rect_contour,
                                                    rect,
                                                    concatenated.shape[1],
                                                    image_height,
                                                    test_contour)
  fig = plt.figure()
  plt.imshow(positional_view_image)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def show_contour(contour, test_contour=None, padding=0,
                 title='medial phalanx variation', minimize_image: bool = True,
                 show_convex_defects: bool = True):
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
                              title=title,
                              test_contour=test_contour,
                              show_convex_defects=show_convex_defects)

def visualize_medial_phalanx_shape():
  borders_detected = Image.open('docs/medial_phalanx.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], padding=5,
               title='medial phalanx original.', minimize_image=False)
  
  under_80_area = np.array(
    [[[ 0,  0]],
    [[ 0, 12]],
    [[ 5, 12]],
    [[ 7, 10]],
    [[ 6,  9]],
    [[ 6,  4]],
    [[ 5,  3]],
    [[ 6,  2]],
    [[ 6,  1]],
    [[ 7,  0]],
    [[ 6,  0]],
    [[ 5,  1]],
    [[ 2,  1]],
    [[ 1,  0]]],
    dtype=np.int32
  )
  show_contour(under_80_area, padding=5, title='Under 80 area medial phalanx.')

  bad_aspect_ratio = np.array(
    [[[0, 0]],
    [[0, 7]],
    [[5, 7]],
    [[6, 6]],
    [[7, 6]],
    [[6, 5]],
    [[6, 4]],
    [[5, 3]],
    [[6, 2]],
    [[6, 1]],
    [[7, 0]],
    [[6, 0]],
    [[5, 1]],
    [[2, 1]],
    [[1, 0]]],
    dtype=np.int32
  )
  show_contour(bad_aspect_ratio, padding=5,
               title='Bad aspect ratio medial phalanx.')

  larger_aspect_contour = np.array(
    [[[ 32,  80]],
    [[ 31,  81]],
    [[ 29,  81]],
    [[ 28,  82]],
    [[ 27,  82]],
    [[ 26,  83]],
    [[ 22,  83]],
    [[ 21,  82]],
    [[ 17,  82]],
    [[ 15,  84]],
    [[ 15,  89]],
    [[ 17,  91]],
    [[ 17,  92]],
    [[ 19,  94]],
    [[ 19,  96]],
    [[ 20,  97]],
    [[ 20, 106]],
    [[ 22, 108]],
    [[ 22, 109]],
    [[ 23, 110]],
    [[ 23, 115]],
    [[ 24, 116]],
    [[ 24, 117]],
    [[ 25, 118]],
    [[ 25, 123]],
    [[ 26, 124]],
    [[ 26, 132]],
    [[ 27, 133]],
    [[ 27, 134]],
    [[ 35, 134]],
    [[ 36, 133]],
    [[ 41, 133]],
    [[ 42, 132]],
    [[ 43, 132]],
    [[ 44, 131]],
    [[ 45, 131]],
    [[ 46, 130]],
    [[ 51, 130]],
    [[ 52, 129]],
    [[ 53, 129]],
    [[ 53, 126]],
    [[ 48, 121]],
    [[ 48, 120]],
    [[ 45, 117]],
    [[ 45, 116]],
    [[ 43, 114]],
    [[ 43, 113]],
    [[ 42, 112]],
    [[ 42, 104]],
    [[ 41, 103]],
    [[ 41, 100]],
    [[ 40,  99]],
    [[ 40,  97]],
    [[ 39,  96]],
    [[ 39,  95]],
    [[ 37,  93]],
    [[ 37,  88]],
    [[ 38,  87]],
    [[ 38,  85]],
    [[ 39,  84]],
    [[ 35,  80]]],
    dtype=np.int32
  )
  show_contour(larger_aspect_contour, padding=5,
               title='Second occurrence non within range medial phalanx aspect' \
               ' ratio\'s.')

  high_solidity = np.array(
    [[[ 22,  99]],
    [[ 20, 101]],
    [[ 20, 106]],
    [[ 22, 108]],
    [[ 22, 109]],
    [[ 23, 110]],
    [[ 23, 115]],
    [[ 24, 116]],
    [[ 24, 117]],
    [[ 25, 118]],
    [[ 25, 123]],
    [[ 26, 124]],
    [[ 26, 132]],
    [[ 27, 133]],
    [[ 27, 134]],
    [[ 35, 134]],
    [[ 36, 133]],
    [[ 41, 133]],
    [[ 42, 132]],
    [[ 43, 132]],
    [[ 44, 131]],
    [[ 45, 131]],
    [[ 46, 130]],
    [[ 51, 130]],
    [[ 52, 129]],
    [[ 53, 129]],
    [[ 53, 126]],
    [[ 48, 121]],
    [[ 48, 120]],
    [[ 45, 117]],
    [[ 45, 116]],
    [[ 43, 114]],
    [[ 41, 114]],
    [[ 40, 113]],
    [[ 37, 113]],
    [[ 36, 112]],
    [[ 35, 112]],
    [[ 34, 111]],
    [[ 34, 109]],
    [[ 33, 108]],
    [[ 33, 103]],
    [[ 32, 102]],
    [[ 32,  99]],
    [[ 31, 100]],
    [[ 27, 100]],
    [[ 26,  99]]],
    dtype=np.int32
  )
  show_contour(high_solidity, padding=5, title='High solidity medial phalanx.')
  
  over_convex_defects = np.array(
    [[[ 22,  99]],
    [[ 20, 101]],
    [[ 20, 106]],
    [[ 22, 108]],
    [[ 22, 109]],
    [[ 23, 110]],
    [[ 23, 115]],
    [[ 24, 116]],
    [[ 24, 117]],
    [[ 25, 118]],
    [[ 25, 123]],
    [[ 26, 124]],
    [[ 26, 132]],
    [[ 27, 133]],
    [[ 27, 134]],
    [[ 35, 134]],
    [[ 36, 133]],
    [[ 41, 133]],
    [[ 42, 132]],
    [[ 43, 132]],
    [[ 44, 131]],
    [[ 45, 131]],
    [[ 46, 130]],
    [[ 51, 130]],
    [[ 52, 129]],
    [[ 53, 129]],
    [[ 53, 126]],
    [[ 52, 125]],
    [[ 43, 125]],
    [[ 42, 124]],
    [[ 43, 123]],
    [[ 43, 122]],
    [[ 45, 120]],
    [[ 45, 119]],
    [[ 46, 118]],
    [[ 45, 117]],
    [[ 45, 116]],
    [[ 43, 114]],
    [[ 41, 114]],
    [[ 40, 113]],
    [[ 37, 113]],
    [[ 36, 112]],
    [[ 35, 112]],
    [[ 34, 111]],
    [[ 34, 109]],
    [[ 33, 108]],
    [[ 33, 103]],
    [[ 32, 102]],
    [[ 32,  99]],
    [[ 31, 100]],
    [[ 27, 100]],
    [[ 26,  99]]],
    dtype=np.int32
  )
  show_contour(over_convex_defects, padding=5,
               title='Too many significant convexity defects medial phalanx.')

  under_convex_defects = np.array(
    [[[ 36,  97]],
    [[ 35,  98]],
    [[ 34,  98]],
    [[ 33,  99]],
    [[ 32,  99]],
    [[ 31, 100]],
    [[ 27, 100]],
    [[ 26,  99]],
    [[ 22,  99]],
    [[ 20, 101]],
    [[ 20, 106]],
    [[ 22, 108]],
    [[ 22, 109]],
    [[ 23, 110]],
    [[ 23, 115]],
    [[ 24, 116]],
    [[ 24, 117]],
    [[ 25, 118]],
    [[ 25, 123]],
    [[ 26, 124]],
    [[ 26, 132]],
    [[ 27, 133]],
    [[ 27, 134]],
    [[ 35, 134]],
    [[ 36, 133]],
    [[ 41, 133]],
    [[ 42, 132]],
    [[ 43, 132]],
    [[ 44, 131]],
    [[ 45, 131]],
    [[ 46, 130]],
    [[ 51, 130]],
    [[ 52, 129]],
    [[ 53, 129]],
    [[ 53, 126]],
    [[ 48, 121]],
    [[ 48, 120]],
    [[ 45, 117]],
    [[ 45, 116]],
    [[ 43, 114]],
    [[ 43, 113]],
    [[ 42, 112]],
    [[ 42, 111]],
    [[ 41, 110]],
    [[ 41, 108]],
    [[ 40, 107]],
    [[ 40, 106]],
    [[ 39, 105]],
    [[ 39, 104]],
    [[ 38, 103]],
    [[ 38, 101]],
    [[ 37, 100]],
    [[ 37,  99]],
    [[ 36,  98]]],
    dtype=np.int32
  )
  show_contour(under_convex_defects, padding=5,
               title='Too few significant convexity defects medial phalanx.')

  plt.show()
