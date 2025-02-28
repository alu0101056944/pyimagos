'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for metacarpal phalanx
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
  if width and height > 0:
    aspect_ratio = max(width, height) / min(width, height)
  else:
    aspect_ratio = 0

  solidity = (width * height) / (
      cv.contourArea(cv.convexHull(contour))
  )

  if show_convex_defects:
    significant_convexity_defects = 0
    hull_area = cv.contourArea(cv.convexHull(contour))
    hull = cv.convexHull(contour, returnPoints=False)
    hull[::-1].sort(axis=0)
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
  hu_moments = np.absolute(hu_moments)
  hu_moments_no_zeros = np.where( # to avoid DivideByZero
    hu_moments == 0,
    np.finfo(float).eps,
    hu_moments
  )
  hu_moments = (np.log10(hu_moments_no_zeros)).flatten()

  reference_hu_moments = np.array(
    [
     -0.37473269,
    -0.84061534,
    -3.91968783,
    -4.34543824,
    -8.4969161,
    -5.00217622,
    -9.01736599
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
    hull_indices[::-1].sort(axis=0)
    defects = cv.convexityDefects(contour, hull_indices)
    if defects is not None:
      for i in range(defects.shape[0]):
        start_index, end_index, farthest_point_index, distance = defects[i, 0]

        start = contour[start_index]
        end = contour[end_index]
        farthest = contour[farthest_point_index]

        defect_area = cv.contourArea(np.array([start, end, farthest]))

        cv.line(hull_defects_image, start, end, (255, 0, 0), 1)
        if defect_area / hull_area > 0.07:
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
                 title='metacarpal variation', minimize_image: bool = True,
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

  epsilon = 0.8 * cv.arcLength(contours[0], closed=True)
  approximated_contour = cv.approxPolyDP(contours[0], epsilon, True)
  approximated_contour = np.reshape(approximated_contour, (-1, 2))

  prepare_image_showing_shape(contours, approximated_contour,
                              image_width=minimal_image.shape[1],
                              image_height=minimal_image.shape[0],
                              title=title,
                              test_contour=test_contour,
                              show_convex_defects=show_convex_defects)

def visualize_metacarpal_shape():
  borders_detected = Image.open('docs/metacarpal_closed.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], padding=5,
               title='metacarpal original.', minimize_image=False)
  
  under_80_area = np.array(
      [[[ 4,  0]],
       [[ 3,  1]],
       [[ 2,  1]],
       [[ 0,  3]],
       [[ 0,  5]],
       [[ 1,  6]],
       [[ 1,  7]],
       [[ 2,  8]],
       [[ 1,  9]],
       [[ 1, 12]],
       [[ 0, 13]],
       [[ 0, 14]],
       [[ 1, 14]],
       [[ 2, 15]],
       [[ 5, 15]],
       [[ 7, 13]],
       [[ 7, 11]],
       [[ 6, 10]],
       [[ 6,  9]],
       [[ 5,  8]],
       [[ 6,  7]],
       [[ 6,  5]],
       [[ 7,  4]],
       [[ 7,  2]],
       [[ 6,  1]],
       [[ 5,  1]]],
    dtype=np.int32
  )
  show_contour(under_80_area, padding=5, title='Under 80 area metacarpal.')

  bad_aspect_ratio = np.array(
    [[[ 29,  75]],
    [[ 28,  76]],
    [[ 24,  76]],
    [[ 23,  77]],
    [[ 21,  77]],
    [[ 20,  78]],
    [[ 19,  78]],
    [[ 18,  79]],
    [[ 17,  79]],
    [[ 15,  81]],
    [[ 15,  82]],
    [[ 14,  83]],
    [[ 14,  84]],
    [[ 15,  85]],
    [[ 15,  89]],
    [[ 14,  90]],
    [[ 14,  93]],
    [[ 19,  98]],
    [[ 20,  98]],
    [[ 21,  99]],
    [[ 21, 100]],
    [[ 22, 101]],
    [[ 22, 102]],
    [[ 23, 103]],
    [[ 23, 104]],
    [[ 24, 105]],
    [[ 24, 106]],
    [[ 25, 107]],
    [[ 25, 108]],
    [[ 26, 109]],
    [[ 26, 110]],
    [[ 27, 111]],
    [[ 27, 112]],
    [[ 28, 113]],
    [[ 28, 114]],
    [[ 29, 115]],
    [[ 29, 116]],
    [[ 30, 117]],
    [[ 30, 119]],
    [[ 31, 120]],
    [[ 31, 123]],
    [[ 32, 124]],
    [[ 32, 128]],
    [[ 33, 129]],
    [[ 33, 138]],
    [[ 34, 139]],
    [[ 34, 141]],
    [[ 35, 142]],
    [[ 35, 143]],
    [[ 38, 146]],
    [[ 40, 146]],
    [[ 41, 147]],
    [[ 44, 147]],
    [[ 45, 148]],
    [[ 47, 148]],
    [[ 48, 147]],
    [[ 49, 147]],
    [[ 50, 146]],
    [[ 50, 145]],
    [[ 52, 143]],
    [[ 52, 142]],
    [[ 54, 140]],
    [[ 54, 139]],
    [[ 57, 136]],
    [[ 57, 135]],
    [[ 59, 133]],
    [[ 59, 132]],
    [[ 61, 130]],
    [[ 61, 129]],
    [[ 62, 128]],
    [[ 60, 126]],
    [[ 60, 125]],
    [[ 59, 125]],
    [[ 50, 116]],
    [[ 50, 115]],
    [[ 48, 113]],
    [[ 48, 112]],
    [[ 46, 110]],
    [[ 46, 107]],
    [[ 45, 106]],
    [[ 45, 101]],
    [[ 44, 100]],
    [[ 44,  91]],
    [[ 43,  90]],
    [[ 43,  87]],
    [[ 42,  86]],
    [[ 42,  84]],
    [[ 34,  76]],
    [[ 32,  76]],
    [[ 31,  75]]],
    dtype=np.int32
  )
  show_contour(bad_aspect_ratio, padding=5,
               title='Bad aspect ratio metacarpal.')

  larger_aspect_contour = np.array(
    [[[ 24,  30]],
       [[ 23,  31]],
       [[ 19,  31]],
       [[ 18,  32]],
       [[ 16,  32]],
       [[ 15,  33]],
       [[ 14,  33]],
       [[ 13,  34]],
       [[ 12,  34]],
       [[ 10,  36]],
       [[ 10,  37]],
       [[  9,  38]],
       [[  9,  39]],
       [[ 10,  40]],
       [[ 10,  44]],
       [[  9,  45]],
       [[  9,  48]],
       [[ 13,  52]],
       [[ 13,  53]],
       [[ 16,  56]],
       [[ 16,  57]],
       [[ 20,  61]],
       [[ 20,  62]],
       [[ 23,  65]],
       [[ 23,  66]],
       [[ 24,  67]],
       [[ 24,  68]],
       [[ 26,  70]],
       [[ 26,  71]],
       [[ 29,  74]],
       [[ 29,  75]],
       [[ 31,  77]],
       [[ 31,  78]],
       [[ 33,  80]],
       [[ 33,  81]],
       [[ 35,  83]],
       [[ 35,  84]],
       [[ 38,  87]],
       [[ 38,  88]],
       [[ 40,  90]],
       [[ 40,  91]],
       [[ 41,  92]],
       [[ 41,  93]],
       [[ 42,  94]],
       [[ 42,  95]],
       [[ 43,  96]],
       [[ 43,  97]],
       [[ 45,  99]],
       [[ 45, 100]],
       [[ 46, 101]],
       [[ 46, 102]],
       [[ 47, 103]],
       [[ 47, 104]],
       [[ 48, 105]],
       [[ 48, 106]],
       [[ 49, 107]],
       [[ 49, 108]],
       [[ 50, 109]],
       [[ 50, 110]],
       [[ 51, 111]],
       [[ 51, 112]],
       [[ 52, 113]],
       [[ 52, 114]],
       [[ 53, 115]],
       [[ 53, 116]],
       [[ 54, 117]],
       [[ 54, 119]],
       [[ 55, 120]],
       [[ 55, 123]],
       [[ 56, 124]],
       [[ 56, 128]],
       [[ 57, 129]],
       [[ 57, 138]],
       [[ 58, 139]],
       [[ 58, 141]],
       [[ 59, 142]],
       [[ 59, 143]],
       [[ 62, 146]],
       [[ 64, 146]],
       [[ 65, 147]],
       [[ 68, 147]],
       [[ 69, 148]],
       [[ 71, 148]],
       [[ 72, 147]],
       [[ 73, 147]],
       [[ 74, 146]],
       [[ 74, 145]],
       [[ 76, 143]],
       [[ 76, 142]],
       [[ 78, 140]],
       [[ 78, 139]],
       [[ 81, 136]],
       [[ 81, 135]],
       [[ 83, 133]],
       [[ 83, 132]],
       [[ 85, 130]],
       [[ 85, 129]],
       [[ 86, 128]],
       [[ 84, 126]],
       [[ 84, 125]],
       [[ 83, 125]],
       [[ 74, 116]],
       [[ 74, 115]],
       [[ 72, 113]],
       [[ 72, 112]],
       [[ 70, 110]],
       [[ 70, 109]],
       [[ 68, 107]],
       [[ 68, 106]],
       [[ 67, 105]],
       [[ 67, 104]],
       [[ 66, 103]],
       [[ 66, 102]],
       [[ 65, 101]],
       [[ 65, 100]],
       [[ 64,  99]],
       [[ 64,  98]],
       [[ 63,  97]],
       [[ 63,  96]],
       [[ 62,  95]],
       [[ 62,  93]],
       [[ 61,  92]],
       [[ 61,  91]],
       [[ 60,  90]],
       [[ 60,  89]],
       [[ 59,  88]],
       [[ 59,  86]],
       [[ 58,  85]],
       [[ 58,  83]],
       [[ 57,  82]],
       [[ 58,  81]],
       [[ 56,  79]],
       [[ 56,  78]],
       [[ 53,  75]],
       [[ 53,  74]],
       [[ 50,  71]],
       [[ 50,  70]],
       [[ 46,  66]],
       [[ 46,  65]],
       [[ 43,  62]],
       [[ 43,  61]],
       [[ 40,  58]],
       [[ 40,  56]],
       [[ 39,  55]],
       [[ 39,  46]],
       [[ 38,  45]],
       [[ 38,  42]],
       [[ 37,  41]],
       [[ 37,  39]],
       [[ 29,  31]],
       [[ 27,  31]],
       [[ 26,  30]]],
    dtype=np.int32
  )
  show_contour(larger_aspect_contour, padding=5,
               title='Second occurrence non within range metacarpal aspect' \
               ' ratio\'s.')

  high_solidity = np.array(
    [[[ 33,  56]],
       [[ 32,  57]],
       [[ 31,  57]],
       [[ 30,  58]],
       [[ 29,  58]],
       [[ 27,  60]],
       [[ 27,  61]],
       [[ 26,  62]],
       [[ 26,  63]],
       [[ 27,  64]],
       [[ 27,  68]],
       [[ 26,  69]],
       [[ 26,  72]],
       [[ 30,  76]],
       [[ 30,  77]],
       [[ 33,  80]],
       [[ 33,  81]],
       [[ 35,  83]],
       [[ 35,  84]],
       [[ 38,  87]],
       [[ 38,  88]],
       [[ 40,  90]],
       [[ 40,  91]],
       [[ 41,  92]],
       [[ 41,  93]],
       [[ 42,  94]],
       [[ 42,  95]],
       [[ 43,  96]],
       [[ 43,  97]],
       [[ 45,  99]],
       [[ 45, 100]],
       [[ 46, 101]],
       [[ 46, 102]],
       [[ 47, 103]],
       [[ 47, 104]],
       [[ 48, 105]],
       [[ 48, 106]],
       [[ 49, 107]],
       [[ 49, 108]],
       [[ 50, 109]],
       [[ 50, 110]],
       [[ 51, 111]],
       [[ 51, 112]],
       [[ 52, 113]],
       [[ 52, 114]],
       [[ 53, 115]],
       [[ 53, 116]],
       [[ 54, 117]],
       [[ 54, 119]],
       [[ 55, 120]],
       [[ 55, 123]],
       [[ 56, 124]],
       [[ 56, 128]],
       [[ 57, 129]],
       [[ 57, 138]],
       [[ 58, 139]],
       [[ 58, 141]],
       [[ 59, 142]],
       [[ 59, 143]],
       [[ 62, 146]],
       [[ 64, 146]],
       [[ 65, 147]],
       [[ 68, 147]],
       [[ 69, 148]],
       [[ 71, 148]],
       [[ 72, 147]],
       [[ 73, 147]],
       [[ 74, 146]],
       [[ 74, 145]],
       [[ 76, 143]],
       [[ 76, 142]],
       [[ 78, 140]],
       [[ 78, 139]],
       [[ 81, 136]],
       [[ 81, 135]],
       [[ 83, 133]],
       [[ 83, 132]],
       [[ 85, 130]],
       [[ 85, 129]],
       [[ 86, 128]],
       [[ 84, 126]],
       [[ 84, 125]],
       [[ 83, 125]],
       [[ 74, 116]],
       [[ 74, 115]],
       [[ 72, 113]],
       [[ 72, 112]],
       [[ 70, 110]],
       [[ 70, 109]],
       [[ 68, 107]],
       [[ 68, 106]],
       [[ 67, 105]],
       [[ 67, 104]],
       [[ 66, 103]],
       [[ 66, 102]],
       [[ 65, 101]],
       [[ 65, 100]],
       [[ 64,  99]],
       [[ 64,  98]],
       [[ 63,  97]],
       [[ 63,  96]],
       [[ 62,  95]],
       [[ 62,  93]],
       [[ 61,  92]],
       [[ 61,  91]],
       [[ 60,  90]],
       [[ 60,  89]],
       [[ 59,  88]],
       [[ 58,  88]],
       [[ 56,  86]],
       [[ 55,  86]],
       [[ 54,  85]],
       [[ 53,  85]],
       [[ 51,  83]],
       [[ 50,  83]],
       [[ 48,  81]],
       [[ 47,  81]],
       [[ 45,  79]],
       [[ 45,  78]],
       [[ 44,  77]],
       [[ 44,  76]],
       [[ 43,  75]],
       [[ 43,  74]],
       [[ 42,  73]],
       [[ 42,  72]],
       [[ 41,  71]],
       [[ 41,  70]],
       [[ 40,  69]],
       [[ 40,  68]],
       [[ 38,  66]],
       [[ 38,  65]],
       [[ 37,  64]],
       [[ 37,  63]],
       [[ 36,  62]],
       [[ 36,  61]],
       [[ 35,  60]],
       [[ 35,  59]],
       [[ 34,  58]],
       [[ 34,  57]]],
    dtype=np.int32
  )
  show_contour(high_solidity, padding=5, title='High solidity metacarpal.')
  
  over_convex_defects = np.array(
   [[[ 41,  54]],
       [[ 40,  55]],
       [[ 36,  55]],
       [[ 35,  56]],
       [[ 33,  56]],
       [[ 32,  57]],
       [[ 31,  57]],
       [[ 30,  58]],
       [[ 29,  58]],
       [[ 27,  60]],
       [[ 27,  61]],
       [[ 26,  62]],
       [[ 26,  63]],
       [[ 27,  64]],
       [[ 27,  68]],
       [[ 26,  69]],
       [[ 26,  72]],
       [[ 29,  75]],
       [[ 37,  75]],
       [[ 38,  76]],
       [[ 46,  76]],
       [[ 47,  77]],
       [[ 46,  78]],
       [[ 46,  79]],
       [[ 45,  80]],
       [[ 45,  82]],
       [[ 44,  83]],
       [[ 44,  84]],
       [[ 43,  85]],
       [[ 43,  87]],
       [[ 42,  88]],
       [[ 42,  89]],
       [[ 40,  91]],
       [[ 40,  92]],
       [[ 39,  93]],
       [[ 39,  94]],
       [[ 37,  96]],
       [[ 37,  97]],
       [[ 36,  98]],
       [[ 36,  99]],
       [[ 35, 100]],
       [[ 36, 101]],
       [[ 36, 102]],
       [[ 37, 103]],
       [[ 37, 104]],
       [[ 39, 106]],
       [[ 39, 107]],
       [[ 41, 109]],
       [[ 41, 110]],
       [[ 42, 111]],
       [[ 42, 112]],
       [[ 43, 113]],
       [[ 52, 113]],
       [[ 53, 112]],
       [[ 62, 112]],
       [[ 63, 113]],
       [[ 62, 114]],
       [[ 62, 116]],
       [[ 61, 117]],
       [[ 61, 119]],
       [[ 60, 120]],
       [[ 60, 123]],
       [[ 59, 124]],
       [[ 59, 126]],
       [[ 58, 127]],
       [[ 58, 129]],
       [[ 57, 130]],
       [[ 57, 138]],
       [[ 58, 139]],
       [[ 58, 141]],
       [[ 59, 142]],
       [[ 59, 143]],
       [[ 62, 146]],
       [[ 64, 146]],
       [[ 65, 147]],
       [[ 68, 147]],
       [[ 69, 148]],
       [[ 71, 148]],
       [[ 72, 147]],
       [[ 73, 147]],
       [[ 74, 146]],
       [[ 74, 145]],
       [[ 76, 143]],
       [[ 76, 142]],
       [[ 78, 140]],
       [[ 78, 139]],
       [[ 81, 136]],
       [[ 81, 135]],
       [[ 83, 133]],
       [[ 83, 132]],
       [[ 85, 130]],
       [[ 85, 129]],
       [[ 86, 128]],
       [[ 84, 126]],
       [[ 84, 125]],
       [[ 83, 125]],
       [[ 74, 116]],
       [[ 74, 115]],
       [[ 72, 113]],
       [[ 72, 112]],
       [[ 70, 110]],
       [[ 70, 109]],
       [[ 68, 107]],
       [[ 68, 106]],
       [[ 67, 105]],
       [[ 67, 104]],
       [[ 66, 103]],
       [[ 66, 102]],
       [[ 65, 101]],
       [[ 64, 101]],
       [[ 63, 100]],
       [[ 62, 100]],
       [[ 61,  99]],
       [[ 60,  99]],
       [[ 59,  98]],
       [[ 57,  98]],
       [[ 56,  97]],
       [[ 55,  97]],
       [[ 54,  96]],
       [[ 53,  96]],
       [[ 51,  94]],
       [[ 52,  93]],
       [[ 52,  92]],
       [[ 53,  91]],
       [[ 53,  89]],
       [[ 54,  88]],
       [[ 54,  87]],
       [[ 55,  86]],
       [[ 55,  84]],
       [[ 56,  83]],
       [[ 56,  82]],
       [[ 57,  81]],
       [[ 57,  80]],
       [[ 56,  79]],
       [[ 56,  70]],
       [[ 55,  69]],
       [[ 55,  66]],
       [[ 54,  65]],
       [[ 54,  63]],
       [[ 46,  55]],
       [[ 44,  55]],
       [[ 43,  54]]],
    dtype=np.int32
  )
  show_contour(over_convex_defects, padding=5,
               title='Too many significant convexity defects metacarpal.')

  under_convex_defects = np.array(
    [[[ 41,  54]],
       [[ 40,  55]],
       [[ 36,  55]],
       [[ 35,  56]],
       [[ 33,  56]],
       [[ 32,  57]],
       [[ 31,  57]],
       [[ 30,  58]],
       [[ 29,  58]],
       [[ 27,  60]],
       [[ 27,  61]],
       [[ 26,  62]],
       [[ 26,  63]],
       [[ 27,  64]],
       [[ 27,  68]],
       [[ 26,  69]],
       [[ 26,  72]],
       [[ 30,  76]],
       [[ 30,  77]],
       [[ 32,  79]],
       [[ 32,  80]],
       [[ 33,  81]],
       [[ 33,  82]],
       [[ 34,  83]],
       [[ 34,  84]],
       [[ 35,  85]],
       [[ 35,  86]],
       [[ 36,  87]],
       [[ 36,  88]],
       [[ 37,  89]],
       [[ 37,  90]],
       [[ 38,  91]],
       [[ 38,  92]],
       [[ 39,  93]],
       [[ 39,  94]],
       [[ 40,  95]],
       [[ 40,  96]],
       [[ 41,  97]],
       [[ 41,  98]],
       [[ 42,  99]],
       [[ 42, 100]],
       [[ 43, 101]],
       [[ 43, 102]],
       [[ 44, 103]],
       [[ 44, 104]],
       [[ 45, 105]],
       [[ 45, 106]],
       [[ 46, 107]],
       [[ 46, 108]],
       [[ 47, 109]],
       [[ 47, 110]],
       [[ 48, 111]],
       [[ 48, 112]],
       [[ 49, 113]],
       [[ 49, 114]],
       [[ 50, 115]],
       [[ 50, 116]],
       [[ 51, 117]],
       [[ 51, 118]],
       [[ 52, 119]],
       [[ 52, 120]],
       [[ 53, 121]],
       [[ 53, 122]],
       [[ 54, 123]],
       [[ 54, 124]],
       [[ 55, 125]],
       [[ 55, 126]],
       [[ 56, 127]],
       [[ 56, 128]],
       [[ 57, 129]],
       [[ 57, 138]],
       [[ 58, 139]],
       [[ 58, 141]],
       [[ 59, 142]],
       [[ 59, 143]],
       [[ 62, 146]],
       [[ 64, 146]],
       [[ 65, 147]],
       [[ 68, 147]],
       [[ 69, 148]],
       [[ 71, 148]],
       [[ 72, 147]],
       [[ 73, 147]],
       [[ 74, 146]],
       [[ 74, 145]],
       [[ 76, 143]],
       [[ 76, 142]],
       [[ 78, 140]],
       [[ 78, 139]],
       [[ 81, 136]],
       [[ 81, 135]],
       [[ 83, 133]],
       [[ 83, 132]],
       [[ 85, 130]],
       [[ 85, 129]],
       [[ 86, 128]],
       [[ 84, 126]],
       [[ 84, 125]],
       [[ 83, 125]],
       [[ 78, 120]],
       [[ 78, 119]],
       [[ 77, 118]],
       [[ 77, 117]],
       [[ 76, 116]],
       [[ 76, 115]],
       [[ 75, 114]],
       [[ 75, 113]],
       [[ 74, 112]],
       [[ 74, 111]],
       [[ 73, 110]],
       [[ 73, 109]],
       [[ 72, 108]],
       [[ 72, 107]],
       [[ 71, 106]],
       [[ 71, 105]],
       [[ 70, 104]],
       [[ 70, 103]],
       [[ 69, 102]],
       [[ 69, 101]],
       [[ 68, 100]],
       [[ 68,  99]],
       [[ 67,  98]],
       [[ 67,  97]],
       [[ 66,  96]],
       [[ 66,  95]],
       [[ 65,  94]],
       [[ 65,  93]],
       [[ 64,  92]],
       [[ 64,  91]],
       [[ 63,  90]],
       [[ 63,  89]],
       [[ 62,  88]],
       [[ 62,  87]],
       [[ 61,  86]],
       [[ 61,  85]],
       [[ 60,  84]],
       [[ 60,  83]],
       [[ 59,  82]],
       [[ 59,  81]],
       [[ 58,  80]],
       [[ 58,  79]],
       [[ 57,  78]],
       [[ 57,  77]],
       [[ 56,  76]],
       [[ 56,  70]],
       [[ 55,  69]],
       [[ 55,  66]],
       [[ 54,  65]],
       [[ 54,  63]],
       [[ 46,  55]],
       [[ 44,  55]],
       [[ 43,  54]]],
       dtype=np.int32
  )
  show_contour(under_convex_defects, padding=5,
               title='Too few significant convexity defects metacarpal.')

  plt.show()
