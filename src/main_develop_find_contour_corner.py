'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Given a specific bone's contour, find a specific corner. This is used
to try to implement a way to characterize and find that corner.
'''

from PIL import Image

import numpy as np
import cv2 as cv

from src.main_develop_contours_gui import (
  draw_contours, draw_contour_points
)

def get_bounding_box_xy(contour):
  x, y, w, h = cv.boundingRect(contour)
  return x, y

def case_1():
  borders_detected = Image.open('docs/radius_bone_minimal.jpg')
  borders_detected = np.array(borders_detected)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_BGR2GRAY)

  _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)

  thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
  contours_drawn_image, contour_colors = draw_contour_points(
    thresh,
    contours,
    draw_points=True
  )
  contours_drawn_image = draw_contours(
    contours_drawn_image,
    contours,
    contour_colors
  )

  contours_drawn_image_scaled = cv.resize(contours_drawn_image, (0, 0), fx=2,
                      fy=2,
                      interpolation=cv.INTER_NEAREST)
  
  cv.imshow('Drawn contours', contours_drawn_image_scaled)
  cv.waitKey(0)
  cv.destroyAllWindows()

def find_contour_corner():
  case_1()
