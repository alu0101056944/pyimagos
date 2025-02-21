'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for radius
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
    defects = cv.convexityDefects(contour, hull)
    if defects is not None:
      for i in range(defects.shape[0]):
        start_index, end_index, farthest_point_index, distance = defects[i, 0]

        start = contour[start_index]
        end = contour[end_index]
        farthest = contour[farthest_point_index]

        defect_area = cv.contourArea(np.array([start, end, farthest]))

        if defect_area / hull_area > 0.005:
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
      -0.66925245,
      -1.84932298,
      -2.72591812,
      -4.1386608,
      -7.86227902,
      -5.16346411,
      -7.63675214,
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
        if defect_area / hull_area > 0.005:
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
                 title='radius variation', minimize_image: bool = True,
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

def visualize_radius_shape():
  borders_detected = Image.open('docs/radius_closed.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], padding=5,
               title='radius original.', minimize_image=False)
  

  low_area = np.array(
    [[[14,  0]],
    [[13,  1]],
    [[ 0,  1]],
    [[ 0,  5]],
    [[ 1,  5]],
    [[ 2,  6]],
    [[ 1,  7]],
    [[ 0,  7]],
    [[ 0, 19]],
    [[10, 19]],
    [[10, 18]],
    [[11, 17]],
    [[11, 16]],
    [[12, 15]],
    [[12, 14]],
    [[13, 13]],
    [[13, 12]],
    [[14, 11]],
    [[14, 10]],
    [[15,  9]],
    [[15,  8]],
    [[14,  7]],
    [[15,  6]],
    [[15,  0]]],
    dtype=np.int32
  )
  show_contour(low_area, padding=5,
               title='Under 200 area radius.')

  aspect_ratio = np.array(
    [[[14,  0]],
    [[13,  1]],
    [[ 0,  1]],
    [[ 0,  3]],
    [[ 1,  4]],
    [[ 1,  5]],
    [[ 0,  6]],
    [[ 0, 15]],
    [[11, 15]],
    [[12, 14]],
    [[12, 13]],
    [[13, 12]],
    [[13, 11]],
    [[14, 10]],
    [[14,  9]],
    [[15,  8]],
    [[14,  7]],
    [[15,  6]],
    [[15,  0]]],
    dtype=np.int32
  )
  show_contour(aspect_ratio, padding=5,
               title='Bad aspect ratio radius.')

  high_solidity = np.array(
    [[[ 0,  0]],
    [[ 0, 17]],
    [[ 1, 16]],
    [[ 2, 16]],
    [[ 3, 15]],
    [[ 4, 15]],
    [[ 6, 13]],
    [[ 7, 13]],
    [[ 8, 12]],
    [[ 9, 12]],
    [[11, 10]],
    [[12, 10]],
    [[13,  9]],
    [[14,  9]],
    [[15,  8]],
    [[14,  7]],
    [[15,  6]],
    [[14,  6]],
    [[13,  5]],
    [[12,  5]],
    [[11,  4]],
    [[ 9,  4]],
    [[ 8,  3]],
    [[ 7,  3]],
    [[ 6,  2]],
    [[ 4,  2]],
    [[ 3,  1]],
    [[ 2,  1]],
    [[ 1,  0]]],
    dtype=np.int32
  )
  show_contour(high_solidity, padding=5,
               title='Solidity too high (radius).')

  overconvex = np.array(
    [[[200, 351]],
    [[199, 352]],
    [[197, 352]],
    [[196, 353]],
    [[194, 353]],
    [[193, 354]],
    [[191, 354]],
    [[190, 355]],
    [[187, 355]],
    [[186, 356]],
    [[182, 356]],
    [[181, 357]],
    [[177, 357]],
    [[176, 358]],
    [[155, 358]],
    [[154, 359]],
    [[151, 359]],
    [[150, 358]],
    [[149, 359]],
    [[138, 359]],
    [[137, 358]],
    [[132, 358]],
    [[131, 359]],
    [[129, 359]],
    [[127, 361]],
    [[127, 364]],
    [[126, 365]],
    [[126, 366]],
    [[129, 369]],
    [[128, 370]],
    [[126, 370]],
    [[125, 371]],
    [[125, 374]],
    [[124, 375]],
    [[124, 377]],
    [[122, 379]],
    [[122, 380]],
    [[123, 381]],
    [[123, 383]],
    [[124, 384]],
    [[124, 400]],
    [[123, 401]],
    [[123, 407]],
    [[122, 408]],
    [[122, 411]],
    [[121, 412]],
    [[121, 415]],
    [[120, 416]],
    [[120, 418]],
    [[119, 419]],
    [[119, 422]],
    [[117, 424]],
    [[117, 428]],
    [[116, 429]],
    [[116, 430]],
    [[114, 432]],
    [[114, 434]],
    [[113, 435]],
    [[113, 436]],
    [[112, 437]],
    [[112, 439]],
    [[111, 440]],
    [[111, 444]],
    [[126, 444]],
    [[127, 443]],
    [[127, 442]],
    [[129, 440]],
    [[129, 439]],
    [[131, 437]],
    [[131, 436]],
    [[133, 434]],
    [[133, 433]],
    [[134, 432]],
    [[136, 434]],
    [[136, 435]],
    [[141, 440]],
    [[141, 441]],
    [[143, 443]],
    [[151, 443]],
    [[152, 442]],
    [[153, 442]],
    [[155, 440]],
    [[155, 439]],
    [[156, 438]],
    [[156, 437]],
    [[158, 435]],
    [[158, 434]],
    [[159, 433]],
    [[159, 432]],
    [[161, 430]],
    [[161, 429]],
    [[162, 428]],
    [[162, 427]],
    [[163, 426]],
    [[163, 425]],
    [[165, 423]],
    [[165, 422]],
    [[166, 421]],
    [[166, 420]],
    [[168, 418]],
    [[168, 417]],
    [[170, 415]],
    [[170, 414]],
    [[173, 411]],
    [[173, 410]],
    [[176, 407]],
    [[176, 406]],
    [[188, 394]],
    [[188, 393]],
    [[192, 389]],
    [[192, 388]],
    [[194, 386]],
    [[194, 384]],
    [[195, 383]],
    [[195, 379]],
    [[196, 378]],
    [[197, 378]],
    [[198, 377]],
    [[199, 377]],
    [[203, 373]],
    [[203, 372]],
    [[205, 370]],
    [[205, 369]],
    [[206, 368]],
    [[206, 367]],
    [[208, 365]],
    [[208, 364]],
    [[209, 363]],
    [[209, 361]],
    [[210, 360]],
    [[210, 358]],
    [[211, 357]],
    [[211, 356]],
    [[210, 355]],
    [[210, 353]],
    [[209, 352]],
    [[208, 352]],
    [[207, 351]]],
    dtype=np.int32
  )
  show_contour(overconvex, padding=5,
               title='Too many defects (radius).')

  underconvex = np.array(
    [[[200, 351]],
    [[199, 352]],
    [[197, 352]],
    [[196, 353]],
    [[194, 353]],
    [[193, 354]],
    [[191, 354]],
    [[190, 355]],
    [[187, 355]],
    [[186, 356]],
    [[182, 356]],
    [[181, 357]],
    [[177, 357]],
    [[176, 358]],
    [[155, 358]],
    [[154, 359]],
    [[151, 359]],
    [[150, 358]],
    [[149, 359]],
    [[138, 359]],
    [[137, 358]],
    [[132, 358]],
    [[131, 359]],
    [[129, 359]],
    [[127, 361]],
    [[127, 364]],
    [[126, 365]],
    [[126, 366]],
    [[129, 369]],
    [[128, 370]],
    [[126, 370]],
    [[125, 371]],
    [[125, 374]],
    [[124, 375]],
    [[124, 377]],
    [[122, 379]],
    [[122, 380]],
    [[123, 381]],
    [[123, 383]],
    [[124, 384]],
    [[124, 400]],
    [[123, 401]],
    [[123, 407]],
    [[122, 408]],
    [[122, 411]],
    [[121, 412]],
    [[121, 415]],
    [[120, 416]],
    [[120, 418]],
    [[119, 419]],
    [[119, 422]],
    [[117, 424]],
    [[117, 428]],
    [[116, 429]],
    [[116, 430]],
    [[114, 432]],
    [[114, 434]],
    [[113, 435]],
    [[113, 436]],
    [[112, 437]],
    [[112, 439]],
    [[111, 440]],
    [[111, 444]],
    [[130, 444]],
    [[131, 443]],
    [[151, 443]],
    [[152, 442]],
    [[153, 442]],
    [[155, 440]],
    [[155, 439]],
    [[156, 438]],
    [[156, 437]],
    [[158, 435]],
    [[158, 434]],
    [[159, 433]],
    [[159, 432]],
    [[161, 430]],
    [[161, 429]],
    [[162, 428]],
    [[162, 427]],
    [[163, 426]],
    [[163, 425]],
    [[165, 423]],
    [[165, 422]],
    [[166, 421]],
    [[166, 420]],
    [[168, 418]],
    [[168, 417]],
    [[170, 415]],
    [[170, 414]],
    [[173, 411]],
    [[173, 410]],
    [[176, 407]],
    [[176, 406]],
    [[188, 394]],
    [[188, 393]],
    [[192, 389]],
    [[192, 388]],
    [[194, 386]],
    [[194, 384]],
    [[197, 381]],
    [[197, 380]],
    [[199, 378]],
    [[199, 377]],
    [[203, 373]],
    [[203, 372]],
    [[205, 370]],
    [[205, 369]],
    [[206, 368]],
    [[206, 367]],
    [[208, 365]],
    [[208, 364]],
    [[209, 363]],
    [[209, 361]],
    [[210, 360]],
    [[210, 358]],
    [[211, 357]],
    [[211, 356]],
    [[210, 355]],
    [[210, 353]],
    [[209, 352]],
    [[208, 352]],
    [[207, 351]]],
    dtype=np.int32
  )
  show_contour(underconvex, padding=5,
               title='Too few defects (radius).')

  plt.show()
