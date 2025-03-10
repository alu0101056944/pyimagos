'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Show the radiography segments with a text over it indicating the tag of the
segment. For example, for distal phalanx 1 show a text over the distal phalanx
segment annotating it as "distal phalanx 1".
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.main_develop_test_distal_phalanx import (
  create_minimal_image_from_contours,
)

from src.radiographies.rad_004 import case_004
from src.radiographies.rad_022 import case_022
from src.radiographies.rad_006 import case_006
from src.radiographies.rad_018 import case_018
from src.radiographies.rad_023 import case_023
from src.radiographies.rad_029 import case_029
from src.radiographies.rad_032 import case_032
from src.radiographies.rad_217 import case_217
from src.radiographies.rad_1622 import case_1622
from src.radiographies.rad_1886 import case_1886
from src.radiographies.rad_010 import case_010
from src.radiographies.rad_013 import case_013
from src.radiographies.rad_016 import case_016
from src.radiographies.rad_019 import case_019
from src.radiographies.rad_030 import case_030
from src.radiographies.rad_031 import case_031
from src.radiographies.rad_084 import case_084
from src.radiographies.rad_1619 import case_1619
from src.radiographies.rad_1779 import case_1779
from src.radiographies.rad_2089 import case_2089

def show_segment_tags(contours, title: str):
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  max_x = int(np.max(x_values))
  max_y = int(np.max(y_values))

  blank_image = np.zeros((max_y + 20, max_x + 20), dtype=np.uint8)
  minimal_image, adjusted_contours = create_minimal_image_from_contours(
    blank_image,
    contours,
    padding=60,
  )
  minimal_image = cv.cvtColor(minimal_image, cv.COLOR_GRAY2RGB)
  contours = adjusted_contours

  for i, contour in enumerate(contours):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(minimal_image, contours, i, color, 1)

    M = cv.moments(contour)
    if M["m00"] != 0: # use centroid
      center_x = int(M["m10"] / M["m00"])
      center_y = int(M["m01"] / M["m00"])
      text_position = (center_x, center_y)
    else: # if too small use bounding box
      x, y, w, h = cv.boundingRect(contour)
      text_position = (x, y - 10)

    cv.putText(
      minimal_image,
      f'{i}',
      text_position,
      cv.FONT_HERSHEY_SIMPLEX,
      fontScale=0.5,
      color=color,
      thickness=1,
      lineType=cv.LINE_AA
    )

  fig = plt.figure()
  plt.imshow(minimal_image)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def visualize_tags_main():
  segments = case_004()
  show_segment_tags(segments, '004 radiography tags.')

  segments = case_022()
  show_segment_tags(segments, '022 radiography tags.')

  segments = case_006()
  show_segment_tags(segments, '006 radiography tags')

  segments = case_018()
  show_segment_tags(segments, '018 radiography tags')

  segments = case_023()
  show_segment_tags(segments, '023 radiography tags')

  segments = case_029()
  show_segment_tags(segments, '029 radiography tags')

  segments = case_032()
  show_segment_tags(segments, '032 radiography tags')

  segments = case_217()
  show_segment_tags(segments, '217 radiography tags')

  segments = case_1622()
  show_segment_tags(segments, '1622 radiography tags')

  segments = case_1886()
  show_segment_tags(segments, '1886 radiography tags')

  segments = case_010() # Cannot estimate, missing distal 4
  show_segment_tags(segments, '010 radiography tags')

  segments = case_013()
  show_segment_tags(segments, '013 radiography tags')

  segments = case_016()
  show_segment_tags(segments, '016 radiography tags')

  segments = case_019()
  show_segment_tags(segments, '019 radiography tags')

  segments = case_030()
  show_segment_tags(segments, '030 radiography tags')

  segments = case_031()
  show_segment_tags(segments, '031 radiography tags')

  segments = case_084()
  show_segment_tags(segments, '084 radiography tags')

  segments = case_1619()
  show_segment_tags(segments, '1619 radiography tags')

  segments = case_1779()
  show_segment_tags(segments, '1779 radiography tags')

  segments = case_2089()
  show_segment_tags(segments, '2089 radiography tags')
  plt.show()
