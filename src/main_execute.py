'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Processing steps of the radiograph
'''

import os.path
from PIL import Image

import torchvision.transforms as transforms
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

def process_radiograph(filename: str, write_images: bool = False,
                       show_images: bool = True) -> None:
  input_image = Image.open(filename)
  input_image = transforms.ToTensor()(input_image)

  he_enchanced = ContrastEnhancement().process(input_image)
  he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  gaussian_blurred = cv.GaussianBlur(he_enchanced, (5, 5), 0)
  borders_detected = cv.Canny(gaussian_blurred, 40, 135)
  borders_detected = cv.normalize(borders_detected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  # Segmentation

  # num_markers, markers = cv.connectedComponents(borders_detected)

  # initialize an origin coordinate (old)
  non_zeros = np.nonzero(borders_detected)
  row_indices = non_zeros[0]
  col_indices = non_zeros[1]
  distances_to_topleft_origin = np.sqrt(row_indices ** 2 + col_indices ** 2)
  min_index = np.argmin(distances_to_topleft_origin)
  origin = (row_indices[min_index], col_indices[min_index])

  AMOUNT_OF_CONTOURS_TO_SAVE = 5

  all_contours = [
    []
  ]

  # old

  # state_stack = [contour]
  # def advance_contour(contour, state_stack):
  #   '''Search and extension of a given contour'''

  #   possible_directions = []
  #   window = 
  #   return new_contour

  # while origin != None:
    # While contour non completely explored:

    #   Apply ruleset to discard/include and calculate a difference value.

    #   Save contour and difference if not discarded

    #   Search next contour or fallback

    # if non all expected contours have been found
    # then update origin to the next relative position
    # or stop if no relative position is possible due to
    # all remaining valued as 0

  if write_images:
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
               f'processed_canny_borders.jpg',
               borders_detected)
  else:
    if show_images:
      cv.imshow(f'{os.path.basename(filename)}' \
                f'processed_canny_borders.jpg',
                borders_detected)
      cv.waitKey(0)
      cv.destroyAllWindows()
