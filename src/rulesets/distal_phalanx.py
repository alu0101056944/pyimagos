'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Rulesets for checking whether a contour is a distal phalanx.
'''

import cv2 as cv

def check_area(contour_image):
  return [False, -1] if  cv.contourArea(contour_image) < 100 else [True, 0]

ruleset = [
  check_area,
]
