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
import copy

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
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.ulna import ExpectedContourUlna
from constants import EXECUTION_DURATION_SECONDS
from src.contour_operations.cut_contour import CutContour
from src.contour_operations.extend_contour import ExtendContour
from src.contour_operations.join_contour import JoinContour

# From left-top to right-bottom of the image (0, 0) to (size x, size y)
distal_phalanx_1 = ExpectedContourDistalPhalanx(
  encounter_amount=1,
)
medial_phalanx_1 = ExpectedContourMedialPhalanx(
  encounter_amount=1,
  first_in_branch=distal_phalanx_1
)
proximal_phalanx_1 = ExpectedContourProximalPhalanx(
  encounter_amount=1,
  first_in_branch=distal_phalanx_1
)
metacarpal_1 = ExpectedContourMetacarpal(
  encounter_amount=1,
  first_in_branch=distal_phalanx_1
)

distal_phalanx_2 = ExpectedContourDistalPhalanx(
  encounter_amount=2,
  first_occurence=distal_phalanx_1
)
medial_phalanx_2 = ExpectedContourMedialPhalanx(
  encounter_amount=2,
  first_occurence=medial_phalanx_1,
  first_in_branch=distal_phalanx_2
)
proximal_phalanx_2 = ExpectedContourProximalPhalanx(
  encounter_amount=2,
  first_occurence=proximal_phalanx_1,
  first_in_branch=distal_phalanx_2
)
metacarpal_2 = ExpectedContourMetacarpal(
  encounter_amount=2,
  first_occurence=metacarpal_1,
  first_in_branch=distal_phalanx_2
)

distal_phalanx_3 = ExpectedContourDistalPhalanx(
  encounter_amount=3,
  first_occurence=distal_phalanx_1,
)
medial_phalanx_3 = ExpectedContourMedialPhalanx(
  encounter_amount=3,
  first_occurence=medial_phalanx_1,
  first_in_branch=distal_phalanx_3
)
proximal_phalanx_3 = ExpectedContourProximalPhalanx(
  encounter_amount=3,
  first_occurence=proximal_phalanx_1,
  first_in_branch=distal_phalanx_3
)
metacarpal_3 = ExpectedContourMetacarpal(
  encounter_amount=3,
  first_occurence=metacarpal_1,
  first_in_branch=distal_phalanx_3
)

distal_phalanx_4 = ExpectedContourDistalPhalanx(
  encounter_amount=4,
  first_occurence=distal_phalanx_1,
)
medial_phalanx_4 = ExpectedContourMedialPhalanx(
  encounter_amount=4,
  first_occurence=medial_phalanx_1,
  first_in_branch=distal_phalanx_4
)
proximal_phalanx_4 = ExpectedContourProximalPhalanx(
  encounter_amount=4,
  first_occurence=proximal_phalanx_1,
  first_in_branch=distal_phalanx_4
)
metacarpal_4 = ExpectedContourMetacarpal(
  encounter_amount=4,
  first_occurence=metacarpal_1,
  first_in_branch=distal_phalanx_4
)

distal_phalanx_5 = ExpectedContourDistalPhalanx(
  encounter_amount=5,
  first_occurence=distal_phalanx_1,
)
proximal_phalanx_5 = ExpectedContourProximalPhalanx(
  encounter_amount=5,
  first_occurence=proximal_phalanx_1,
  first_in_branch=distal_phalanx_5
)
metacarpal_5 = ExpectedContourMetacarpal(
  encounter_amount=5,
  first_occurence=metacarpal_1,
  first_in_branch=distal_phalanx_5,
  ends_branchs_sequence=True
)

radius = ExpectedContourRadius()
ulna = ExpectedContourUlna()

expected_contours = [
  distal_phalanx_1,
  medial_phalanx_1,
  proximal_phalanx_1,
  metacarpal_1,
  distal_phalanx_2,
  medial_phalanx_2,
  proximal_phalanx_2,
  metacarpal_2,
  distal_phalanx_3,
  medial_phalanx_3,
  proximal_phalanx_3,
  metacarpal_3,
  distal_phalanx_4,
  medial_phalanx_4,
  proximal_phalanx_4,
  metacarpal_4,
  distal_phalanx_5,
  proximal_phalanx_5,
  metacarpal_5,
  radius,
  ulna,
]

def find_closest_contours_to_point(point: np.array, contours: list) -> np.array:
  contour_distances = np.array([
    np.min(
      np.sqrt(np.sum((contour - point) ** 2, axis=1))
    ) for contour in contours
  ])

  sorted_index = np.argsort(contour_distances)
  return sorted_index

def is_in_allowed_space(contour: list,
  last_expected_contour: ExpectedContour
) -> bool:
  position_restrictions = last_expected_contour.next_contour_restrictions()
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
                                  reference_state: dict, image_width: int,
                                  image_height: int) -> list:
  alternatives = []

  contour = contours[contour_id]
  for i in range(len(contour)):
    cut_operation = CutContour(contour_id, i, image_width, image_height)
    new_contour = cut_operation.generate_new_contour(contours)
    state = dict(reference_state)
    state['contours_committed'] = copy.deepcopy(reference_state)
    state['contours'] = new_contour
    state['contours_alternatives'] = []
    state['has_generated_alternatives'] = False
    alternatives.append(state)
  
  invasion_counts = [1, 2, 3, 4]
  for i in range(len(contours)):
    for j in range(len(contours)):
      for invasion_count in invasion_counts:
        extend_operation = ExtendContour(
          i,
          j,
          image_width,
          image_height,
          invasion_count
        )
        new_contour = extend_operation.generate_new_contour(contours)
        state = dict(reference_state)
        state['contours_committed'] = copy.deepcopy(reference_state)
        state['contours'] = new_contour
        state['contours_alternatives'] = []
        state['has_generated_alternatives'] = False,
        alternatives.append(state)

      join_operation = JoinContour(i, j)
      new_contour = join_operation.generate_new_contour(contours)
      state = dict(reference_state)
      state['contours_committed'] = copy.deepcopy(reference_state)
      state['contours'] = new_contour
      state['contours_alternatives'] = []
      state['has_generated_alternatives'] = False,
      alternatives.append(state)

  return alternatives

def min_contour_distance(contour1: list, contour2: list) -> float:
  distances = np.sqrt(
        np.sum((contour1[:, None] - contour2) ** 2, axis=2)
    )  # contour1[:, None] adds dim, making it column for broadcasting purpose
  min_distance = np.min(distances)
  return min_distance

def search_complete_contours(initial_state_stack: dict,
                             search_duration_seconds: int,
                             image_width: int,
                             image_height: int) -> list:
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

      expected_contour_class.prepare(
        current_contour,
        image_width,
        image_height
      )

      restrictions = expected_contour_class.shape_restrictions()
      current_contour_valid, current_contour_value = restrictions

      if current_contour_valid:
        if len(state['contours_committed']) == len(expected_contours) - 1:
          complete_contours.append(copy.deepcopy(
            [
              [*state['contours_committed'], current_contour],
              state['committed_total_value'] + current_contour_value
            ]
          ))
        else:
          contours_inside_area = filter(
            lambda contour : (
              is_in_allowed_space(contour, expected_contour_class)
            ),
            contours
          )
          if len(contours_inside_area) > 0:
            new_state = dict(state)
            new_state['contours_committed'] = copy.deepcopy(
              state['contours_committed']
            )
            new_state['contours_committed'].append(current_contour)
            new_state['contours'] = copy.deepcopy(contours)
            new_state['contours_alternatives'] = []
            new_state['has_generated_alternatives'] = False,
            new_state['committed_total_value'] = (
              new_state['committed_total_value'] + current_contour_value
            )

            distances = [
              min_contour_distance(
                current_contour,
                contour2
              ) for contour2 in contours_inside_area
            ]
            sorted_indices = np.argsort(distances)
            new_state['current_contour_index'] = sorted_indices[0]

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

def create_minimal_image_from_contours(contours: list) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = np.min(x_values)
  min_y = np.min(y_values)
  max_x = np.max(x_values)
  max_y = np.max(y_values)

  width = max_x - min_x + 1
  height = max_y - min_y + 1

  image = np.zeros((height, width, 3), dtype=np.uint8)

  translated_contours = []
  for contour in contours:
    translated_contour = contour.copy()
    translated_contour[:, 0] -= min_x
    translated_contour[:, 1] -= min_y
    translated_contours.append(translated_contour)

  return image

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
  
  minimize_image_size = create_minimal_image_from_contours(contours)

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

  visualizer = ContourViewer(minimize_image_size, [best_contour])

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
               minimize_image_size)
  else:
    if show_images:
      cv.imshow(f'{os.path.basename(filename)}' \
                f'processed_canny_borders.jpg',
                minimize_image_size)
      cv.waitKey(0)
      cv.destroyAllWindows()
