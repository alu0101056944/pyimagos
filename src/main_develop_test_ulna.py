'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the shape testing for ulna
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
                 title='ulna variation', minimize_image: bool = True,
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

def visualize_ulna_shape():
  try:
    with Image.open('docs/ulna_closed.jpg') as image:
      if image.mode == 'L':
        image = image.convert('RGB')
        borders_detected = np.array(image)
      elif image.mode == 'RGB':
        borders_detected = np.array(image)
  except Exception as e:
    print(f"Error opening image {'docs/ulna_closed.jpg'}: {e}")
    raise
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  show_contour(contours[0], padding=5,
               title='ulna original.', minimize_image=False)
  

  low_area = np.array(
    [[[ 0,  0]],
    [[ 0,  3]],
    [[ 2,  5]],
    [[ 1,  6]],
    [[ 0,  6]],
    [[ 0, 15]],
    [[ 7, 15]],
    [[ 7, 13]],
    [[ 6, 12]],
    [[ 6,  9]],
    [[ 7,  8]],
    [[ 7,  1]],
    [[ 5,  1]],
    [[ 4,  2]],
    [[ 2,  2]]],
    dtype=np.int32
  )
  show_contour(low_area, padding=5,
               title='Under 250 area ulna.')

  aspect_ratio = np.array(
    [[[ 59, 385]],
    [[ 58, 386]],
    [[ 57, 386]],
    [[ 52, 391]],
    [[ 52, 392]],
    [[ 51, 393]],
    [[ 50, 393]],
    [[ 49, 394]],
    [[ 49, 399]],
    [[ 45, 403]],
    [[ 45, 404]],
    [[ 44, 405]],
    [[ 44, 407]],
    [[ 43, 408]],
    [[ 43, 409]],
    [[ 42, 410]],
    [[ 42, 412]],
    [[ 41, 413]],
    [[ 41, 414]],
    [[ 40, 415]],
    [[ 40, 417]],
    [[ 39, 418]],
    [[ 39, 419]],
    [[ 38, 420]],
    [[ 38, 423]],
    [[ 37, 424]],
    [[ 37, 426]],
    [[ 36, 427]],
    [[ 36, 428]],
    [[ 35, 429]],
    [[ 35, 430]],
    [[ 33, 432]],
    [[ 33, 433]],
    [[ 31, 435]],
    [[ 31, 436]],
    [[ 30, 437]],
    [[ 30, 438]],
    [[ 29, 439]],
    [[ 29, 440]],
    [[ 27, 442]],
    [[ 27, 444]],
    [[ 59, 444]],
    [[ 59, 443]],
    [[ 62, 440]],
    [[ 62, 439]],
    [[ 64, 437]],
    [[ 64, 435]],
    [[ 65, 434]],
    [[ 65, 433]],
    [[ 71, 427]],
    [[ 72, 427]],
    [[ 78, 421]],
    [[ 80, 421]],
    [[ 81, 420]],
    [[ 82, 420]],
    [[ 84, 418]],
    [[ 84, 414]],
    [[ 86, 412]],
    [[ 86, 411]],
    [[ 87, 410]],
    [[ 88, 410]],
    [[ 89, 409]],
    [[ 89, 404]],
    [[ 88, 403]],
    [[ 88, 402]],
    [[ 87, 401]],
    [[ 85, 401]],
    [[ 84, 400]],
    [[ 81, 400]],
    [[ 80, 399]],
    [[ 77, 399]],
    [[ 76, 398]],
    [[ 72, 398]],
    [[ 71, 397]],
    [[ 70, 397]],
    [[ 68, 395]],
    [[ 68, 394]],
    [[ 67, 393]],
    [[ 67, 386]],
    [[ 66, 385]]],
    dtype=np.int32
  )
  show_contour(aspect_ratio, padding=5,
               title='Bad aspect ratio ulna.')

  high_solidity = np.array(
    [[[ 87, 342]],
    [[ 86, 343]],
    [[ 85, 343]],
    [[ 80, 348]],
    [[ 80, 349]],
    [[ 79, 350]],
    [[ 78, 350]],
    [[ 77, 351]],
    [[ 77, 356]],
    [[ 73, 360]],
    [[ 73, 361]],
    [[ 72, 362]],
    [[ 72, 364]],
    [[ 71, 365]],
    [[ 71, 366]],
    [[ 70, 367]],
    [[ 70, 369]],
    [[ 69, 370]],
    [[ 69, 371]],
    [[ 68, 372]],
    [[ 68, 374]],
    [[ 67, 375]],
    [[ 67, 376]],
    [[ 66, 377]],
    [[ 66, 378]],
    [[ 65, 379]],
    [[ 65, 380]],
    [[ 64, 381]],
    [[ 64, 382]],
    [[ 63, 383]],
    [[ 63, 384]],
    [[ 62, 385]],
    [[ 62, 386]],
    [[ 61, 387]],
    [[ 61, 388]],
    [[ 59, 390]],
    [[ 59, 391]],
    [[ 58, 392]],
    [[ 58, 393]],
    [[ 57, 394]],
    [[ 57, 395]],
    [[ 55, 397]],
    [[ 55, 398]],
    [[ 53, 400]],
    [[ 53, 401]],
    [[ 52, 402]],
    [[ 52, 403]],
    [[ 50, 405]],
    [[ 50, 406]],
    [[ 48, 408]],
    [[ 48, 409]],
    [[ 46, 411]],
    [[ 46, 412]],
    [[ 45, 413]],
    [[ 45, 414]],
    [[ 43, 416]],
    [[ 43, 417]],
    [[ 41, 419]],
    [[ 41, 420]],
    [[ 40, 421]],
    [[ 40, 422]],
    [[ 37, 425]],
    [[ 37, 426]],
    [[ 36, 427]],
    [[ 36, 428]],
    [[ 35, 429]],
    [[ 35, 430]],
    [[ 33, 432]],
    [[ 33, 433]],
    [[ 31, 435]],
    [[ 31, 436]],
    [[ 30, 437]],
    [[ 30, 438]],
    [[ 29, 439]],
    [[ 29, 440]],
    [[ 27, 442]],
    [[ 27, 444]],
    [[ 31, 444]],
    [[ 34, 441]],
    [[ 35, 441]],
    [[ 42, 434]],
    [[ 43, 434]],
    [[ 50, 427]],
    [[ 51, 427]],
    [[ 58, 420]],
    [[ 59, 420]],
    [[ 66, 413]],
    [[ 67, 413]],
    [[ 74, 406]],
    [[ 75, 406]],
    [[ 82, 399]],
    [[ 83, 399]],
    [[ 90, 392]],
    [[ 91, 392]],
    [[ 98, 385]],
    [[ 99, 385]],
    [[104, 380]],
    [[105, 380]],
    [[107, 378]],
    [[108, 378]],
    [[109, 377]],
    [[110, 377]],
    [[112, 375]],
    [[112, 371]],
    [[114, 369]],
    [[114, 368]],
    [[115, 367]],
    [[116, 367]],
    [[117, 366]],
    [[117, 361]],
    [[116, 360]],
    [[116, 359]],
    [[115, 358]],
    [[113, 358]],
    [[112, 357]],
    [[109, 357]],
    [[108, 356]],
    [[105, 356]],
    [[104, 355]],
    [[100, 355]],
    [[ 99, 354]],
    [[ 98, 354]],
    [[ 96, 352]],
    [[ 96, 351]],
    [[ 95, 350]],
    [[ 95, 343]],
    [[ 94, 342]]],
    dtype=np.int32
  )
  show_contour(high_solidity, padding=5,
               title='Solidity too high (ulna).')

  overconvex = np.array(
   [[[ 87, 342]],
  [[ 86, 343]],
  [[ 85, 343]],
  [[ 80, 348]],
  [[ 80, 349]],
  [[ 79, 350]],
  [[ 78, 350]],
  [[ 77, 351]],
  [[ 77, 356]],
  [[ 73, 360]],
  [[ 73, 361]],
  [[ 72, 362]],
  [[ 72, 364]],
  [[ 71, 365]],
  [[ 71, 366]],
  [[ 70, 367]],
  [[ 70, 369]],
  [[ 69, 370]],
  [[ 69, 371]],
  [[ 68, 372]],
  [[ 68, 374]],
  [[ 67, 375]],
  [[ 67, 376]],
  [[ 66, 377]],
  [[ 66, 378]],
  [[ 65, 379]],
  [[ 65, 380]],
  [[ 64, 381]],
  [[ 64, 382]],
  [[ 63, 383]],
  [[ 63, 384]],
  [[ 62, 385]],
  [[ 62, 386]],
  [[ 61, 387]],
  [[ 61, 388]],
  [[ 59, 390]],
  [[ 59, 391]],
  [[ 58, 392]],
  [[ 58, 393]],
  [[ 57, 394]],
  [[ 57, 395]],
  [[ 55, 397]],
  [[ 55, 398]],
  [[ 53, 400]],
  [[ 53, 401]],
  [[ 52, 402]],
  [[ 52, 403]],
  [[ 50, 405]],
  [[ 50, 406]],
  [[ 48, 408]],
  [[ 48, 409]],
  [[ 46, 411]],
  [[ 46, 412]],
  [[ 45, 413]],
  [[ 45, 414]],
  [[ 43, 416]],
  [[ 43, 417]],
  [[ 41, 419]],
  [[ 41, 420]],
  [[ 40, 421]],
  [[ 40, 422]],
  [[ 37, 425]],
  [[ 37, 426]],
  [[ 36, 427]],
  [[ 36, 428]],
  [[ 35, 429]],
  [[ 35, 430]],
  [[ 33, 432]],
  [[ 33, 433]],
  [[ 31, 435]],
  [[ 31, 436]],
  [[ 30, 437]],
  [[ 30, 438]],
  [[ 29, 439]],
  [[ 29, 440]],
  [[ 27, 442]],
  [[ 27, 444]],
  [[ 32, 444]],
  [[ 32, 443]],
  [[ 33, 442]],
  [[ 34, 442]],
  [[ 37, 439]],
  [[ 38, 439]],
  [[ 41, 436]],
  [[ 42, 436]],
  [[ 45, 433]],
  [[ 46, 433]],
  [[ 48, 431]],
  [[ 51, 434]],
  [[ 51, 435]],
  [[ 56, 440]],
  [[ 56, 441]],
  [[ 59, 444]],
  [[ 59, 443]],
  [[ 62, 440]],
  [[ 62, 439]],
  [[ 64, 437]],
  [[ 64, 435]],
  [[ 65, 434]],
  [[ 65, 433]],
  [[ 66, 432]],
  [[ 66, 431]],
  [[ 68, 429]],
  [[ 68, 428]],
  [[ 69, 427]],
  [[ 69, 426]],
  [[ 70, 425]],
  [[ 70, 424]],
  [[ 71, 423]],
  [[ 71, 421]],
  [[ 72, 420]],
  [[ 72, 419]],
  [[ 73, 418]],
  [[ 73, 415]],
  [[ 75, 413]],
  [[ 75, 411]],
  [[ 76, 410]],
  [[ 76, 409]],
  [[ 78, 407]],
  [[ 78, 406]],
  [[ 79, 405]],
  [[ 79, 404]],
  [[ 81, 402]],
  [[ 81, 401]],
  [[ 83, 399]],
  [[ 83, 398]],
  [[ 95, 386]],
  [[ 96, 386]],
  [[ 99, 383]],
  [[100, 383]],
  [[102, 381]],
  [[103, 381]],
  [[104, 380]],
  [[105, 380]],
  [[107, 378]],
  [[108, 378]],
  [[109, 377]],
  [[110, 377]],
  [[112, 375]],
  [[112, 371]],
  [[114, 369]],
  [[114, 368]],
  [[115, 367]],
  [[116, 367]],
  [[117, 366]],
  [[117, 361]],
  [[116, 360]],
  [[116, 359]],
  [[115, 358]],
  [[113, 358]],
  [[112, 357]],
  [[109, 357]],
  [[108, 356]],
  [[105, 356]],
  [[104, 355]],
  [[100, 355]],
  [[ 99, 354]],
  [[ 98, 354]],
  [[ 96, 352]],
  [[ 96, 351]],
  [[ 95, 350]],
  [[ 95, 343]],
  [[ 94, 342]]],
    dtype=np.int32
  )
  show_contour(overconvex, padding=5,
               title='Too many defects (ulna).')

  underconvex = np.array(
    [[[ 87, 342]],
    [[ 86, 343]],
    [[ 85, 343]],
    [[ 80, 348]],
    [[ 80, 349]],
    [[ 79, 350]],
    [[ 78, 350]],
    [[ 77, 351]],
    [[ 77, 356]],
    [[ 73, 360]],
    [[ 73, 361]],
    [[ 72, 362]],
    [[ 72, 364]],
    [[ 71, 365]],
    [[ 71, 366]],
    [[ 70, 367]],
    [[ 70, 369]],
    [[ 69, 370]],
    [[ 69, 371]],
    [[ 68, 372]],
    [[ 68, 374]],
    [[ 67, 375]],
    [[ 67, 376]],
    [[ 66, 377]],
    [[ 66, 378]],
    [[ 65, 379]],
    [[ 65, 380]],
    [[ 64, 381]],
    [[ 64, 382]],
    [[ 63, 383]],
    [[ 63, 384]],
    [[ 62, 385]],
    [[ 62, 386]],
    [[ 61, 387]],
    [[ 61, 388]],
    [[ 59, 390]],
    [[ 59, 391]],
    [[ 58, 392]],
    [[ 58, 393]],
    [[ 57, 394]],
    [[ 57, 395]],
    [[ 55, 397]],
    [[ 55, 398]],
    [[ 53, 400]],
    [[ 53, 401]],
    [[ 52, 402]],
    [[ 52, 403]],
    [[ 50, 405]],
    [[ 50, 406]],
    [[ 48, 408]],
    [[ 48, 409]],
    [[ 46, 411]],
    [[ 46, 412]],
    [[ 45, 413]],
    [[ 45, 414]],
    [[ 43, 416]],
    [[ 43, 417]],
    [[ 41, 419]],
    [[ 41, 420]],
    [[ 40, 421]],
    [[ 40, 422]],
    [[ 37, 425]],
    [[ 37, 426]],
    [[ 36, 427]],
    [[ 36, 428]],
    [[ 35, 429]],
    [[ 35, 430]],
    [[ 33, 432]],
    [[ 33, 433]],
    [[ 31, 435]],
    [[ 31, 436]],
    [[ 30, 437]],
    [[ 30, 438]],
    [[ 29, 439]],
    [[ 29, 440]],
    [[ 27, 442]],
    [[ 27, 444]],
    [[ 59, 444]],
    [[ 59, 443]],
    [[ 62, 440]],
    [[ 62, 439]],
    [[ 64, 437]],
    [[ 64, 435]],
    [[ 65, 434]],
    [[ 65, 433]],
    [[ 66, 432]],
    [[ 66, 431]],
    [[ 68, 429]],
    [[ 68, 428]],
    [[ 69, 427]],
    [[ 69, 426]],
    [[ 70, 425]],
    [[ 70, 424]],
    [[ 71, 423]],
    [[ 71, 421]],
    [[ 72, 420]],
    [[ 72, 419]],
    [[ 73, 418]],
    [[ 73, 415]],
    [[ 75, 413]],
    [[ 75, 411]],
    [[ 76, 410]],
    [[ 76, 409]],
    [[ 78, 407]],
    [[ 78, 406]],
    [[ 79, 405]],
    [[ 79, 404]],
    [[ 81, 402]],
    [[ 81, 401]],
    [[ 83, 399]],
    [[ 83, 398]],
    [[ 95, 386]],
    [[ 96, 386]],
    [[ 99, 383]],
    [[100, 383]],
    [[102, 381]],
    [[103, 381]],
    [[104, 380]],
    [[105, 380]],
    [[107, 378]],
    [[108, 378]],
    [[109, 377]],
    [[110, 377]],
    [[112, 375]],
    [[112, 371]],
    [[114, 369]],
    [[114, 368]],
    [[115, 367]],
    [[116, 367]],
    [[117, 366]],
    [[117, 361]],
    [[116, 360]],
    [[116, 359]],
    [[115, 358]],
    [[114, 358]],
    [[112, 356]],
    [[111, 356]],
    [[109, 354]],
    [[108, 354]],
    [[106, 352]],
    [[105, 352]],
    [[102, 349]],
    [[101, 349]],
    [[ 99, 347]],
    [[ 98, 347]],
    [[ 96, 345]],
    [[ 95, 345]],
    [[ 93, 343]],
    [[ 92, 343]],
    [[ 91, 342]]],
    dtype=np.int32
  )
  show_contour(underconvex, padding=5,
               title='Too few defects (ulna).')

  plt.show()
