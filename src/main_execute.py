'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Processing steps of the radiograph
'''

import os.path
from PIL import Image
import time
import random

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from src.main_develop_contours_gui import ContourViewer
from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.expected_contours.expected_contour import (
  ExpectedContour, AllowedLineSideBasedOnYorXOnVertical
)
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.proximal_phalanx import ExpectedContourProximalPhalanx
from constants import EXECUTION_DURATION_SECONDS

# From left-top to right-bottom of the image (0, 0) to (size x, size y)
distal_phalanx_1 = ExpectedContourDistalPhalanx()
medial_phalanx_1 = ExpectedContourMedialPhalanx()
proximal_phalanx_1 = ExpectedContourProximalPhalanx(distal_phalanx_1)

expected_contours = [
  distal_phalanx_1,
  medial_phalanx_1,
  proximal_phalanx_1
]

def find_closest_contours_to_point(point: np.array, contours: list) -> np.array:
  contour_distances = np.array([
    np.min(
      np.sqrt(np.sum((contour - point) ** 2, axis=1))
    ) for contour in contours
  ])

  sorted_index = np.argsort(contour_distances)
  return sorted_index

def is_in_allowed_space(
  contour: list,
  last_expected_contour: list[ExpectedContour]
) -> bool:
  position_restrictions = last_expected_contour.position_restrictions()
  for position_restriction in position_restrictions:
    x1, y1 = position_restriction[0][0]
    x2, y2 = position_restriction[1][0]
    allowed_side = position_restriction[2]

    if x2 == x1:
      # Vertical line case handling
      for point in contour:
        x, y = point[0]
        if allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER:
          if x <= x1:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
          if x >= x1:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
          if x < x1:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
          if x > x1:
            return False
    else:
      # line equation y = mx + b
      m = (y2 - y1) / (x2 - x1)
      b = y1 - m * x1

      for point in contour:
        x, y = point[0]
        line_y = m * x + b
        if allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER:
          if y <= line_y:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
          if y >= line_y:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
          if y < line_y:
            return False
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
          if y > line_y:
            return False
  return True

def generate_contour_alternatives(contours: list, contour_id: int, 
                                  reference_state: dict) -> list:
  pass

def search_complete_contours(initial_state_stack: dict,
                         search_duration_seconds: int) -> list:
  complete_contours = []

  state_stack = initial_state_stack
  start_time = time.time()
  while True:
    elapsed_time = time.time() - start_time
    if elapsed_time >= search_duration_seconds:
      break
    print(f'Elapsed time: {elapsed_time:.2f} seconds')

    if len(state_stack) > 0:
      state = state_stack[0]
      contours = state['contours']
      current_contour_index = state['current_contour_index']
      current_contour = contours[current_contour_index]

      expected_contour_class = (
        expected_contours[len(state['contours_committed'])]
      )
      expected_contour_class.prepare(current_contour)

      restrictions = expected_contour_class.shape_restrictions()
      current_contour_valid, current_contour_value = restrictions

      if current_contour_valid:
        # cálculo de siguiente contorno en base a restricciones de posición
        expected_contours_until_now = (
          expected_contours[:len(state['contours_committed']) + 1]
        )
        contours_within_allowed_space = (
          filter(lambda contour : (
            is_in_allowed_space(contour, expected_contours_until_now)
          ))
        )
        # TODO Falta estudiar cómo hacer que se posicione bien al lado
        # Hay algo que me salté: si bajo del distal al medio y al próximo
        # voy eliminando área global en la imagen y entonces los ...
        # Estudiar bien cómo hacerlo.

    # TODO REVISAR ESTO
        if len(contours_within_allowed_space) == 0:
          if len(state['contours_committed']) + 1 == len(expected_contours):
            complete_contours.append(
              [
                [*state['contours_committed'], current_contour],
                state['committed_total_value'] + current_contour_value
              ]
            )
        
            new_state = list(state)
            new_state['contours_committed'].append(current_contour)
            
            # new_state['current_contour_index'] = <contorno que más sentido tiene>
            new_state['committed_total_value'] = (
              new_state['committed_total_value'] + current_contour_value
            )
            state['contours_alternatives'].append(new_state)

      if not state['has_generated_alternatives']:
        contours_alternatives = generate_contour_alternatives(
          contours,
          current_contour_index,
          state
        )
        state['contours_alternatives'].extend(contours_alternatives)
        state['has_generated_alternatives'] = True

      if len(state['contours_alternatives']) == 0:
        state_stack.pop(0)
      else:
        # Choose alternative
        # TODO Dont just choose random alternative.
        random_index = random.randnt(0, len(state['contours_alternatives'] - 1))
        state_stack.insert(0, state['contours_alternatives'][random_index])
        state['contours_alternatives'].pop(random_index)
    else:
      print('Search finished: explored all alternatives')
      break

  return complete_contours

def process_radiograph(filename: str, write_images: bool = False,
                       show_images: bool = True) -> None:
  global expected_contours
  if len(expected_contours) == 0:
     raise ValueError("The global list 'expected_contours' must have at least one' \
                      ' element to work correctly.")

  input_image = Image.open(filename)
  input_image = transforms.ToTensor()(input_image)

  # TODO add cli flag to signal that input image is borders_detected

  # Temporarily skip contrast enhancement as it takes a long time
  # while developing. Directly input the HE image.
  # TODO turn this back on
  # he_enchanced = ContrastEnhancement().process(input_image)
  # he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX,
  #                             cv.CV_8U)

  # gaussian_blurred = cv.GaussianBlur(he_enchanced, (5, 5), 0)
  gaussian_blurred = cv.GaussianBlur(input_image, (5, 5), 0)
  borders_detected = cv.Canny(gaussian_blurred, 40, 135)
  borders_detected = cv.normalize(borders_detected, None, 0, 255, cv.NORM_MINMAX,
                                  cv.CV_8U)

  # Segmentation
  contours, _ = cv.findContours(borders_detected, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)

  current_contour_index = find_closest_contours_to_point([[0, 0]], contours)[0]

  state_stack = [{
    'contours_committed': [],
    'contours': contours,
    'contours_alternatives': [],
    'current_contour_index': current_contour_index,
    'has_generated_alternatives': False,
    'committed_total_value': 0
  }]

  complete_contours = search_complete_contours(state_stack,
                                               EXECUTION_DURATION_SECONDS)
  complete_contours.sort(key=lambda item: item[1], reverse=True)
  best_contour = complete_contours[0]

  visualizer = ContourViewer(borders_detected, [best_contour])

  # Measurements processing

  # Thumb proximal phalanx padding and intersesamoid search on a
  # more detailed image. Calculate distance from intersesamoid
  # to proximal phalanx. Also width and height of proximal phalanx
  # by counting pixels. At the end, calculate relative distances
  # of each pixel measurement and compare with relative distances
  # atlas to get the final age.

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
