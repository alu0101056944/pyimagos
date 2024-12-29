'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

GUI for contour work visualization:
  - Zoom the image to see a specific contour's points
  - See step history of contour modification
'''

import os.path
from PIL import Image

import numpy as np
import cv2 as cv

from src.image_filters.contrast_enhancement import ContrastEnhancement
import src.rulesets.distal_phalanx as dist_phalanx

expected_contours = [
  {
    'relative_distance': (0, 0),
    'ruleset': dist_phalanx.ruleset
  }
]

def draw_contour_points(image: np.array, contours: np.array) -> np.array:
  colored_image = np.copy(image)

  for i, contour in enumerate(contours):
    for point in contour:
      color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)

      b, g, r = color
      x = point[0][0]
      y = point[0][1]
      cv.circle(colored_image, (x, y), 1, (b, g, r), -1)

  return colored_image

def visualize_contours(filename: str) -> None:
  input_image = Image.open(filename)
  input_image = np.asarray(input_image)
  gaussian_blurred = cv.GaussianBlur(input_image, (5, 5), 0)

  borders_detected = cv.Canny(gaussian_blurred, 40, 135)
  borders_detected = cv.normalize(borders_detected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  # num_markers, markers = cv.connectedComponents(borders_detected)

  contours, _ = cv.findContours(borders_detected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  image_color = cv.cvtColor(borders_detected, cv.COLOR_GRAY2BGR)
  image_color = draw_contour_points(image_color, contours)
  cv.imshow("Image with Contour Points", image_color)
  cv.waitKey(0)
  cv.destroyAllWindows()
