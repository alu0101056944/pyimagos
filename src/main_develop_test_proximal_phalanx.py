'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for proximal phalanx
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
      -0.5707188,
      -1.44230313,
      -2.69970318,
      -3.64888372,
      -6.90187266,
      -4.47148141,
      -7.08174012
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
                 title='proximal phalanx variation', minimize_image: bool = True,
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

def visualize_proximal_phalanx_shape():
  borders_detected = Image.open('docs/proximal_phalanx.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], padding=5,
               title='proximal phalanx original.', minimize_image=False)
  
  under_10_area = np.array(
      [[[0, 0]],
      [[0, 2]],
      [[1, 3]],
      [[0, 4]],
      [[3, 4]],
      [[3, 3]],
      [[2, 2]],
      [[3, 1]],
      [[3, 0]]],
    dtype=np.int32
  )
  show_contour(under_10_area, padding=5, title='Under 10 area proximal phalanx.')

  bad_aspect_ratio = np.array(
    [[[ 50,  92]],
    [[ 48,  94]],
    [[ 46,  94]],
    [[ 45,  95]],
    [[ 43,  95]],
    [[ 42,  96]],
    [[ 34,  96]],
    [[ 33,  97]],
    [[ 32,  97]],
    [[ 32, 105]],
    [[ 35, 108]],
    [[ 35, 109]],
    [[ 37, 111]],
    [[ 37, 112]],
    [[ 38, 113]],
    [[ 38, 115]],
    [[ 39, 114]],
    [[ 40, 115]],
    [[ 40, 116]],
    [[ 41, 117]],
    [[ 41, 118]],
    [[ 42, 119]],
    [[ 42, 121]],
    [[ 41, 122]],
    [[ 41, 125]],
    [[ 40, 126]],
    [[ 40, 130]],
    [[ 39, 131]],
    [[ 39, 138]],
    [[ 41, 140]],
    [[ 49, 140]],
    [[ 50, 139]],
    [[ 51, 139]],
    [[ 52, 138]],
    [[ 54, 138]],
    [[ 55, 137]],
    [[ 57, 137]],
    [[ 58, 136]],
    [[ 61, 136]],
    [[ 62, 135]],
    [[ 73, 135]],
    [[ 74, 134]],
    [[ 75, 134]],
    [[ 75, 132]],
    [[ 76, 131]],
    [[ 76, 129]],
    [[ 75, 128]],
    [[ 75, 127]],
    [[ 74, 126]],
    [[ 73, 126]],
    [[ 58, 111]],
    [[ 58, 110]],
    [[ 57, 109]],
    [[ 57, 104]],
    [[ 56, 103]],
    [[ 56, 101]],
    [[ 57, 100]],
    [[ 57,  99]],
    [[ 56,  98]],
    [[ 56,  96]],
    [[ 55,  95]],
    [[ 54,  95]],
    [[ 51,  92]]],
    dtype=np.int32
  )
  show_contour(bad_aspect_ratio, padding=5,
               title='Bad aspect ratio proximal phalanx.')

  larger_aspect_contour = np.array(
    [[[ 33,  59]],
    [[ 31,  61]],
    [[ 29,  61]],
    [[ 28,  62]],
    [[ 26,  62]],
    [[ 25,  63]],
    [[ 17,  63]],
    [[ 16,  64]],
    [[ 15,  64]],
    [[ 15,  72]],
    [[ 18,  75]],
    [[ 18,  76]],
    [[ 21,  79]],
    [[ 21,  80]],
    [[ 22,  81]],
    [[ 22,  82]],
    [[ 23,  83]],
    [[ 23,  84]],
    [[ 24,  85]],
    [[ 24,  86]],
    [[ 25,  87]],
    [[ 25,  88]],
    [[ 26,  89]],
    [[ 26,  90]],
    [[ 27,  91]],
    [[ 27,  92]],
    [[ 28,  93]],
    [[ 28,  94]],
    [[ 30,  96]],
    [[ 30,  97]],
    [[ 32,  99]],
    [[ 32, 100]],
    [[ 33, 101]],
    [[ 33, 103]],
    [[ 34, 104]],
    [[ 34, 106]],
    [[ 35, 107]],
    [[ 35, 109]],
    [[ 36, 110]],
    [[ 36, 111]],
    [[ 37, 112]],
    [[ 37, 113]],
    [[ 38, 114]],
    [[ 38, 117]],
    [[ 39, 118]],
    [[ 39, 121]],
    [[ 40, 122]],
    [[ 40, 129]],
    [[ 39, 130]],
    [[ 39, 137]],
    [[ 41, 139]],
    [[ 49, 139]],
    [[ 50, 138]],
    [[ 51, 138]],
    [[ 52, 137]],
    [[ 54, 137]],
    [[ 55, 136]],
    [[ 57, 136]],
    [[ 58, 135]],
    [[ 61, 135]],
    [[ 62, 134]],
    [[ 73, 134]],
    [[ 74, 133]],
    [[ 75, 133]],
    [[ 75, 131]],
    [[ 76, 130]],
    [[ 76, 128]],
    [[ 75, 127]],
    [[ 75, 126]],
    [[ 74, 125]],
    [[ 73, 125]],
    [[ 60, 112]],
    [[ 60, 111]],
    [[ 58, 109]],
    [[ 58, 108]],
    [[ 57, 107]],
    [[ 57, 106]],
    [[ 55, 104]],
    [[ 55, 103]],
    [[ 54, 102]],
    [[ 54, 101]],
    [[ 53, 100]],
    [[ 53,  99]],
    [[ 52,  98]],
    [[ 52,  96]],
    [[ 50,  94]],
    [[ 50,  91]],
    [[ 49,  90]],
    [[ 49,  89]],
    [[ 47,  87]],
    [[ 47,  86]],
    [[ 46,  85]],
    [[ 46,  84]],
    [[ 44,  82]],
    [[ 44,  81]],
    [[ 42,  79]],
    [[ 42,  78]],
    [[ 40,  76]],
    [[ 40,  71]],
    [[ 39,  70]],
    [[ 39,  68]],
    [[ 40,  67]],
    [[ 40,  66]],
    [[ 39,  65]],
    [[ 39,  63]],
    [[ 38,  62]],
    [[ 37,  62]],
    [[ 34,  59]]],
    dtype=np.int32
  )
  show_contour(larger_aspect_contour, padding=5,
               title='Second occurrence non within range proximal phalanx aspect' \
               ' ratio\'s.')

  high_solidity = np.array(
    [[[ 36,  80]],
    [[ 35,  81]],
    [[ 27,  81]],
    [[ 26,  82]],
    [[ 25,  82]],
    [[ 25,  90]],
    [[ 28,  93]],
    [[ 28,  94]],
    [[ 30,  96]],
    [[ 30,  97]],
    [[ 32,  99]],
    [[ 32, 100]],
    [[ 33, 101]],
    [[ 33, 103]],
    [[ 34, 104]],
    [[ 34, 106]],
    [[ 35, 107]],
    [[ 35, 109]],
    [[ 36, 110]],
    [[ 36, 111]],
    [[ 37, 112]],
    [[ 37, 113]],
    [[ 38, 114]],
    [[ 38, 117]],
    [[ 39, 118]],
    [[ 39, 121]],
    [[ 40, 122]],
    [[ 40, 129]],
    [[ 39, 130]],
    [[ 39, 137]],
    [[ 41, 139]],
    [[ 49, 139]],
    [[ 50, 138]],
    [[ 51, 138]],
    [[ 52, 137]],
    [[ 54, 137]],
    [[ 55, 136]],
    [[ 57, 136]],
    [[ 58, 135]],
    [[ 61, 135]],
    [[ 62, 134]],
    [[ 73, 134]],
    [[ 74, 133]],
    [[ 75, 133]],
    [[ 75, 131]],
    [[ 76, 130]],
    [[ 76, 128]],
    [[ 75, 127]],
    [[ 75, 126]],
    [[ 74, 125]],
    [[ 73, 125]],
    [[ 60, 112]],
    [[ 60, 111]],
    [[ 57, 108]],
    [[ 56, 108]],
    [[ 54, 106]],
    [[ 53, 106]],
    [[ 52, 105]],
    [[ 51, 105]],
    [[ 49, 103]],
    [[ 48, 103]],
    [[ 46, 101]],
    [[ 45, 101]],
    [[ 44, 100]],
    [[ 44,  99]],
    [[ 43,  98]],
    [[ 43,  96]],
    [[ 42,  95]],
    [[ 42,  93]],
    [[ 41,  92]],
    [[ 41,  90]],
    [[ 40,  89]],
    [[ 40,  88]],
    [[ 39,  87]],
    [[ 39,  85]],
    [[ 38,  84]],
    [[ 38,  82]],
    [[ 37,  81]],
    [[ 37,  80]]],
    dtype=np.int32
  )
  show_contour(high_solidity, padding=5, title='High solidity proximal phalanx.')
  
  over_convex_defects = np.array(
    [[[ 43,  77]],
    [[ 41,  79]],
    [[ 39,  79]],
    [[ 38,  80]],
    [[ 36,  80]],
    [[ 35,  81]],
    [[ 27,  81]],
    [[ 26,  82]],
    [[ 25,  82]],
    [[ 25,  90]],
    [[ 27,  92]],
    [[ 28,  92]],
    [[ 30,  94]],
    [[ 32,  94]],
    [[ 33,  95]],
    [[ 34,  95]],
    [[ 35,  96]],
    [[ 37,  96]],
    [[ 39,  98]],
    [[ 38,  99]],
    [[ 38, 101]],
    [[ 37, 102]],
    [[ 37, 103]],
    [[ 36, 104]],
    [[ 36, 108]],
    [[ 35, 109]],
    [[ 35, 110]],
    [[ 34, 111]],
    [[ 34, 112]],
    [[ 32, 114]],
    [[ 32, 115]],
    [[ 30, 117]],
    [[ 30, 118]],
    [[ 31, 119]],
    [[ 31, 120]],
    [[ 34, 120]],
    [[ 35, 119]],
    [[ 39, 119]],
    [[ 40, 120]],
    [[ 49, 120]],
    [[ 50, 121]],
    [[ 52, 121]],
    [[ 53, 122]],
    [[ 51, 124]],
    [[ 50, 124]],
    [[ 44, 130]],
    [[ 43, 130]],
    [[ 39, 134]],
    [[ 39, 137]],
    [[ 41, 139]],
    [[ 49, 139]],
    [[ 50, 138]],
    [[ 51, 138]],
    [[ 52, 137]],
    [[ 54, 137]],
    [[ 55, 136]],
    [[ 57, 136]],
    [[ 58, 135]],
    [[ 61, 135]],
    [[ 62, 134]],
    [[ 73, 134]],
    [[ 74, 133]],
    [[ 75, 133]],
    [[ 75, 131]],
    [[ 76, 130]],
    [[ 76, 128]],
    [[ 75, 127]],
    [[ 75, 126]],
    [[ 74, 125]],
    [[ 73, 125]],
    [[ 60, 112]],
    [[ 60, 111]],
    [[ 58, 109]],
    [[ 58, 108]],
    [[ 57, 107]],
    [[ 57, 106]],
    [[ 55, 104]],
    [[ 55, 103]],
    [[ 54, 102]],
    [[ 54, 101]],
    [[ 53, 100]],
    [[ 53,  99]],
    [[ 52,  98]],
    [[ 52,  96]],
    [[ 50,  94]],
    [[ 50,  89]],
    [[ 49,  88]],
    [[ 49,  86]],
    [[ 50,  85]],
    [[ 50,  84]],
    [[ 49,  83]],
    [[ 49,  81]],
    [[ 48,  80]],
    [[ 47,  80]],
    [[ 44,  77]]],
    dtype=np.int32
  )
  show_contour(over_convex_defects, padding=5,
               title='Too many significant convexity defects proximal phalanx.')

  under_convex_defects = np.array(
    [[[ 43,  77]],
    [[ 41,  79]],
    [[ 39,  79]],
    [[ 38,  80]],
    [[ 36,  80]],
    [[ 35,  81]],
    [[ 27,  81]],
    [[ 26,  82]],
    [[ 25,  82]],
    [[ 25,  90]],
    [[ 27,  92]],
    [[ 27,  93]],
    [[ 28,  94]],
    [[ 28,  97]],
    [[ 29,  98]],
    [[ 29, 100]],
    [[ 30, 101]],
    [[ 30, 104]],
    [[ 31, 105]],
    [[ 31, 107]],
    [[ 32, 108]],
    [[ 32, 111]],
    [[ 33, 112]],
    [[ 33, 114]],
    [[ 34, 115]],
    [[ 34, 118]],
    [[ 35, 119]],
    [[ 35, 121]],
    [[ 36, 122]],
    [[ 36, 125]],
    [[ 37, 126]],
    [[ 37, 128]],
    [[ 38, 129]],
    [[ 38, 132]],
    [[ 39, 133]],
    [[ 39, 137]],
    [[ 41, 139]],
    [[ 49, 139]],
    [[ 50, 138]],
    [[ 51, 138]],
    [[ 52, 137]],
    [[ 54, 137]],
    [[ 55, 136]],
    [[ 57, 136]],
    [[ 58, 135]],
    [[ 61, 135]],
    [[ 62, 134]],
    [[ 73, 134]],
    [[ 74, 133]],
    [[ 75, 133]],
    [[ 75, 131]],
    [[ 76, 130]],
    [[ 76, 128]],
    [[ 75, 127]],
    [[ 75, 126]],
    [[ 74, 125]],
    [[ 73, 125]],
    [[ 60, 112]],
    [[ 60, 111]],
    [[ 58, 109]],
    [[ 58, 108]],
    [[ 57, 107]],
    [[ 57, 106]],
    [[ 55, 104]],
    [[ 55, 103]],
    [[ 54, 102]],
    [[ 54, 101]],
    [[ 53, 100]],
    [[ 53,  99]],
    [[ 52,  98]],
    [[ 52,  96]],
    [[ 50,  94]],
    [[ 50,  89]],
    [[ 49,  88]],
    [[ 49,  86]],
    [[ 50,  85]],
    [[ 50,  84]],
    [[ 49,  83]],
    [[ 49,  81]],
    [[ 48,  80]],
    [[ 47,  80]],
    [[ 44,  77]]],
    dtype=np.int32
  )
  show_contour(under_convex_defects, padding=5,
               title='Too few significant convexity defects proximal phalanx.',
               show_convex_defects=False)

  open_contour = np.array(
    [[[ 26,  77]],
    [[ 24,  79]],
    [[ 22,  79]],
    [[ 21,  80]],
    [[ 19,  80]],
    [[ 18,  81]],
    [[ 10,  81]],
    [[  9,  82]],
    [[  8,  82]],
    [[  8,  90]],
    [[ 11,  93]],
    [[ 11,  94]],
    [[ 13,  96]],
    [[ 13,  97]],
    [[ 15,  99]],
    [[ 15, 100]],
    [[ 16, 101]],
    [[ 16, 103]],
    [[ 17, 104]],
    [[ 17, 106]],
    [[ 18, 107]],
    [[ 18, 109]],
    [[ 19, 110]],
    [[ 19, 111]],
    [[ 20, 112]],
    [[ 20, 113]],
    [[ 21, 114]],
    [[ 21, 117]],
    [[ 22, 118]],
    [[ 22, 121]],
    [[ 23, 122]],
    [[ 23, 129]],
    [[ 22, 130]],
    [[ 22, 137]],
    [[ 24, 139]],
    [[ 32, 139]],
    [[ 33, 138]],
    [[ 34, 138]],
    [[ 35, 137]],
    [[ 37, 137]],
    [[ 38, 136]],
    [[ 40, 136]],
    [[ 41, 135]],
    [[ 44, 135]],
    [[ 45, 134]],
    [[ 56, 134]],
    [[ 57, 133]],
    [[ 58, 133]],
    [[ 58, 131]],
    [[ 59, 130]],
    [[ 59, 128]],
    [[ 58, 127]],
    [[ 58, 126]],
    [[ 57, 125]],
    [[ 56, 125]],
    [[ 43, 112]],
    [[ 43, 111]],
    [[ 41, 109]],
    [[ 41, 108]],
    [[ 40, 107]],
    [[ 40, 106]],
    [[ 39, 106]],
    [[ 40, 107]],
    [[ 40, 108]],
    [[ 41, 109]],
    [[ 41, 110]],
    [[ 43, 112]],
    [[ 43, 113]],
    [[ 55, 125]],
    [[ 56, 125]],
    [[ 58, 127]],
    [[ 58, 128]],
    [[ 59, 129]],
    [[ 59, 130]],
    [[ 58, 131]],
    [[ 58, 132]],
    [[ 57, 133]],
    [[ 56, 133]],
    [[ 55, 134]],
    [[ 45, 134]],
    [[ 44, 135]],
    [[ 41, 135]],
    [[ 40, 136]],
    [[ 38, 136]],
    [[ 37, 137]],
    [[ 35, 137]],
    [[ 34, 138]],
    [[ 33, 138]],
    [[ 32, 139]],
    [[ 25, 139]],
    [[ 22, 136]],
    [[ 22, 130]],
    [[ 23, 129]],
    [[ 23, 122]],
    [[ 22, 121]],
    [[ 22, 118]],
    [[ 21, 117]],
    [[ 21, 114]],
    [[ 20, 113]],
    [[ 20, 112]],
    [[ 19, 111]],
    [[ 19, 109]],
    [[ 18, 108]],
    [[ 18, 107]],
    [[ 17, 106]],
    [[ 17, 104]],
    [[ 16, 103]],
    [[ 16, 100]],
    [[ 15,  99]],
    [[ 15,  98]],
    [[ 13,  96]],
    [[ 13,  95]],
    [[ 11,  93]],
    [[ 11,  92]],
    [[  8,  89]],
    [[  8,  83]],
    [[ 10,  81]],
    [[ 19,  81]],
    [[ 20,  80]],
    [[ 23,  80]],
    [[ 25,  78]],
    [[ 27,  78]],
    [[ 29,  80]],
    [[ 30,  80]],
    [[ 32,  82]],
    [[ 32,  83]],
    [[ 33,  84]],
    [[ 33,  85]],
    [[ 32,  86]],
    [[ 32,  88]],
    [[ 33,  89]],
    [[ 33,  94]],
    [[ 34,  95]],
    [[ 34,  96]],
    [[ 35,  97]],
    [[ 35,  99]],
    [[ 36, 100]],
    [[ 36, 101]],
    [[ 37, 101]],
    [[ 36, 100]],
    [[ 36,  99]],
    [[ 35,  98]],
    [[ 35,  96]],
    [[ 33,  94]],
    [[ 33,  89]],
    [[ 32,  88]],
    [[ 32,  86]],
    [[ 33,  85]],
    [[ 33,  84]],
    [[ 32,  83]],
    [[ 32,  81]],
    [[ 31,  80]],
    [[ 30,  80]],
    [[ 27,  77]]],
    dtype=np.int32
  )
  show_contour(open_contour, padding=5,
              title='Open contour proximal phalanx.',
              show_convex_defects=False)
  
  small_open = np.array([[[ 26,  77]],
       [[ 24,  79]],
       [[ 22,  79]],
       [[ 21,  80]],
       [[ 19,  80]],
       [[ 18,  81]],
       [[ 10,  81]],
       [[  9,  82]],
       [[  8,  82]],
       [[  8,  90]],
       [[ 11,  93]],
       [[ 11,  94]],
       [[ 13,  96]],
       [[ 13,  97]],
       [[ 15,  99]],
       [[ 15, 100]],
       [[ 16, 101]],
       [[ 16, 103]],
       [[ 17, 104]],
       [[ 17, 106]],
       [[ 18, 107]],
       [[ 18, 109]],
       [[ 19, 110]],
       [[ 19, 111]],
       [[ 20, 112]],
       [[ 20, 113]],
       [[ 21, 114]],
       [[ 21, 117]],
       [[ 22, 118]],
       [[ 22, 121]],
       [[ 23, 122]],
       [[ 23, 129]],
       [[ 22, 130]],
       [[ 22, 137]],
       [[ 24, 139]],
       [[ 32, 139]],
       [[ 33, 138]],
       [[ 34, 138]],
       [[ 35, 137]],
       [[ 37, 137]],
       [[ 38, 136]],
       [[ 40, 136]],
       [[ 41, 135]],
       [[ 44, 135]],
       [[ 45, 134]],
       [[ 56, 134]],
       [[ 57, 133]],
       [[ 58, 133]],
       [[ 58, 131]],
       [[ 59, 130]],
       [[ 59, 128]],
       [[ 58, 127]],
       [[ 58, 126]],
       [[ 57, 125]],
       [[ 56, 125]],
       [[ 43, 112]],
       [[ 43, 111]],
       [[ 41, 109]],
       [[ 41, 108]],
       [[ 40, 107]],
       [[ 40, 106]],
       [[ 39, 106]],
       [[ 38, 105]],
       [[ 38, 103]],
       [[ 37, 103]],
       [[ 38, 104]],
       [[ 38, 106]],
       [[ 39, 106]],
       [[ 40, 107]],
       [[ 40, 108]],
       [[ 41, 109]],
       [[ 41, 110]],
       [[ 43, 112]],
       [[ 43, 113]],
       [[ 55, 125]],
       [[ 56, 125]],
       [[ 58, 127]],
       [[ 58, 128]],
       [[ 59, 129]],
       [[ 59, 130]],
       [[ 58, 131]],
       [[ 58, 132]],
       [[ 57, 133]],
       [[ 56, 133]],
       [[ 55, 134]],
       [[ 45, 134]],
       [[ 44, 135]],
       [[ 41, 135]],
       [[ 40, 136]],
       [[ 38, 136]],
       [[ 37, 137]],
       [[ 35, 137]],
       [[ 34, 138]],
       [[ 33, 138]],
       [[ 32, 139]],
       [[ 25, 139]],
       [[ 22, 136]],
       [[ 22, 130]],
       [[ 23, 129]],
       [[ 23, 122]],
       [[ 22, 121]],
       [[ 22, 118]],
       [[ 21, 117]],
       [[ 21, 114]],
       [[ 20, 113]],
       [[ 20, 112]],
       [[ 19, 111]],
       [[ 19, 109]],
       [[ 18, 108]],
       [[ 18, 107]],
       [[ 17, 106]],
       [[ 17, 104]],
       [[ 16, 103]],
       [[ 16, 100]],
       [[ 15,  99]],
       [[ 15,  98]],
       [[ 13,  96]],
       [[ 13,  95]],
       [[ 11,  93]],
       [[ 11,  92]],
       [[  8,  89]],
       [[  8,  83]],
       [[ 10,  81]],
       [[ 19,  81]],
       [[ 20,  80]],
       [[ 23,  80]],
       [[ 25,  78]],
       [[ 27,  78]],
       [[ 29,  80]],
       [[ 30,  80]],
       [[ 32,  82]],
       [[ 32,  83]],
       [[ 33,  84]],
       [[ 33,  85]],
       [[ 32,  86]],
       [[ 32,  88]],
       [[ 33,  89]],
       [[ 33,  94]],
       [[ 34,  95]],
       [[ 34,  96]],
       [[ 35,  97]],
       [[ 35,  99]],
       [[ 36, 100]],
       [[ 36, 101]],
       [[ 37, 101]],
       [[ 36, 100]],
       [[ 36,  99]],
       [[ 35,  98]],
       [[ 35,  96]],
       [[ 33,  94]],
       [[ 33,  89]],
       [[ 32,  88]],
       [[ 32,  86]],
       [[ 33,  85]],
       [[ 33,  84]],
       [[ 32,  83]],
       [[ 32,  81]],
       [[ 31,  80]],
       [[ 30,  80]],
       [[ 27,  77]]],
       dtype=np.int32
  )
  show_contour(open_contour, padding=5,
            title='Open contour (smaller gap) proximal phalanx.',
            show_convex_defects=False)

  plt.show()
