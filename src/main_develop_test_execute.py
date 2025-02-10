'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the search execution test
'''

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.main_develop_test_distal_phalanx import (
  create_minimal_image_from_contours,
)
from src.main_execute import is_in_allowed_space
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.expected_contour import ExpectedContour

def show_contours_position_restrictions(contours, contour_id,
                                        expected_contour: ExpectedContour, 
                                        padding=5,
                                        title='Position restrictions visualization',
                                        minimize_image: bool = True):
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  max_x = int(np.max(x_values))
  max_y = int(np.max(y_values))

  blank_image = np.zeros((max_y + 300, max_x + 25), dtype=np.uint8)

  if minimize_image:
    minimal_image, adjusted_contours = create_minimal_image_from_contours(
      blank_image,
      contours,
      padding
    )
    minimal_image = cv.cvtColor(minimal_image, cv.COLOR_GRAY2RGB)
    contours = adjusted_contours
  else:
    minimal_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2RGB)

  for i, contour in enumerate(contours):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(minimal_image, contours, i, color, 1)

  expected_contour.prepare(contours[contour_id], minimal_image.shape[1],
                           minimal_image.shape[0])
  position_restrictions = expected_contour.next_contour_restrictions()
  for position_restriction in position_restrictions:
    point_a = position_restriction[0]
    point_b = position_restriction[1]
    direction = point_b - point_a
    cv.line(
      minimal_image,
      (point_a - direction * minimal_image.shape[1]).astype(np.int32),
      (point_b + direction * minimal_image.shape[1]).astype(np.int32),
      (255, 255, 0),
      1
    )
  start_point = expected_contour.orientation_line[0]
  end_point = expected_contour.orientation_line[1]
  cv.line(minimal_image, start_point, end_point, (0, 255, 255), 1)

  fig = plt.figure()
  plt.imshow(minimal_image)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def visualize_execute_tests():
  # borders_detected = Image.open('docs/composition_two.jpg')
  # borders_detected = np.array(borders_detected)
  # borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  # _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  # contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
  #                             cv.CHAIN_APPROX_SIMPLE)

  contours = [
    np.array([[[ 37,  97]],
      [[ 36,  98]],
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
      [[ 42, 105]],
      [[ 43, 104]],
      [[ 43, 102]],
      [[ 44, 101]],
      [[ 40,  97]]],
      dtype=np.int32
    ),
    np.array([[[25, 66]],
      [[24, 67]],
      [[21, 67]],
      [[19, 69]],
      [[19, 72]],
      [[20, 73]],
      [[20, 74]],
      [[21, 75]],
      [[21, 77]],
      [[22, 78]],
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
      [[31, 75]],
      [[32, 74]],
      [[32, 68]],
      [[31, 67]],
      [[28, 67]],
      [[27, 66]]],
      dtype=np.int32
    )
  ]

  expected_contour = ExpectedContourDistalPhalanx(1)
  show_contours_position_restrictions(contours, 1, expected_contour, padding=5,
               title='Composition.', minimize_image=False)
  contours_inside_area = list(filter(
    lambda contour : (
      is_in_allowed_space(contour, expected_contour)
    ),
    contours[:1] + contours[1 + 1:]
  ))
  if len(contours_inside_area) > 0:
    print('Second contour is within expected position.')
  plt.show()