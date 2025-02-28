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
import copy
from typing import Union, Tuple

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.proximal_phalanx import ExpectedContourProximalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.sesamoid import ExpectedContourSesamoid
from src.expected_contours.metacarpal_sesamoid import (
  ExpectedContourSesamoidMetacarpal
)
from src.expected_contours.expected_contour import (
  ExpectedContour, AllowedLineSideBasedOnYorXOnVertical
)
from src.expected_contours.expected_contour_of_branch import (
  ExpectedContourOfBranch
)
from src.expected_contours.sesamoid import (
  ExpectedContourSesamoid
)
# from src.contour_operations.cut_contour import CutContour
# from src.contour_operations.extend_contour import ExtendContour
# from src.contour_operations.join_contour import JoinContour
from constants import (
  SEARCH_EXECUTION_DURATION_SECONDS,
  BONE_AGE_ATLAS,
)

def find_closest_contours_to_point(point: np.array, contours: list) -> np.array:
  contour_distances = np.array([
    np.min(
      np.sqrt(np.sum((contour - point) ** 2, axis=1))
    ) for contour in contours
  ])

  sorted_index = np.argsort(contour_distances)
  return sorted_index

def is_in_allowed_space(contour: list,
  last_expected_contour: ExpectedContour,
  has_to_jump_to_next_branch: bool = False,
  first_in_branch: ExpectedContour = None,
) -> bool:
  if not has_to_jump_to_next_branch:
    position_restrictions = last_expected_contour.next_contour_restrictions()
  else:
    position_restrictions = first_in_branch.branch_start_position_restrictions()

  for position_restriction in position_restrictions:
    x1, y1 = position_restriction[0]
    x2, y2 = position_restriction[1]
    allowed_side_array = position_restriction[2]

    if x2 == x1:
      allowed_side = allowed_side_array[3]
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

      if m > 0:
        allowed_side = allowed_side_array[0]
      elif m < 0:
        allowed_side = allowed_side_array[1]
      else:
        allowed_side = allowed_side_array[2]

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

def get_best_contour_alternative(contours: list, inside_indices: list,
                                 reference_state: dict, expected_contours: list,
                                 image_width: int, image_height: int,
                                 criteria_dict: dict = None) -> list:
  alternatives = []

  for i in inside_indices:
    state = dict(reference_state)
    state['contours_committed'] = list(reference_state['contours_committed'])
    state['contours_committed'].append(contours[i])
    state['contours'] = contours
    state['chosen_contour_index'] = i
    instance = expected_contours[len(state['contours_committed']) - 1]
    instance.prepare(contours[i], image_width, image_height)
    score = instance.shape_restrictions(criteria_dict)
    state['committed_total_value'] = state['committed_total_value'] + score
    alternatives.append(state)

  # for i in inside_indices:
  #   for j in range(len(contours[i])):
  #     cut_operation = CutContour(i, j, image_width, image_height)
  #     new_contours = cut_operation.generate_new_contour(contours)
  #     if new_contours is not None:
  #       state = dict(reference_state)
  #       state['contours_committed'] = list(reference_state['contours_committed'])
  #       state['contours_committed'].append(new_contours[i])
  #       state['contours'] = new_contours
  #       state['chosen_contour_index'] = i
  #       instance = expected_contours[len(state['contours_committed']) - 1]
  #       instance.prepare(new_contours[i], image_width, image_height)
  #       score = instance.shape_restrictions()
  #       state['committed_total_value'] = state['committed_total_value'] + score
  #       alternatives.append(state)

  #       state = dict(reference_state)
  #       state['contours_committed'] = list(reference_state['contours_committed'])
  #       state['contours_committed'].append(new_contours[i])
  #       state['contours'] = new_contours
  #       state['chosen_contour_index'] = len(new_contours) - 1
  #       instance = expected_contours[len(state['contours_committed']) - 1]
  #       instance.prepare(new_contours[len(new_contours) - 1], image_width,
  #                        image_height)
  #       score = instance.shape_restrictions()
  #       state['committed_total_value'] = state['committed_total_value'] + score
  #       alternatives.append(state)
  
  # invasion_counts = [1, 2, 3, 4]
  # for i in inside_indices:
  #   for j in inside_indices:
  #     if i != j:
  #       for invasion_count in invasion_counts:
  #         extend_operation = ExtendContour(
  #           i,
  #           j,
  #           image_width,
  #           image_height,
  #           invasion_count
  #         )
  #         new_contours = extend_operation.generate_new_contour(contours)
  #         if new_contours is not None:
  #           state = dict(reference_state)
  #           state['contours_committed'] = list(reference_state['contours_committed'])
  #           state['contours'] = new_contours
  #           state['chosen_contour_index'] = i
  #           instance = expected_contours[len(state['contours_committed']) - 1]
  #           instance.prepare(new_contours[i], image_width, image_height)
  #           score = instance.shape_restrictions()
  #           state['committed_total_value'] = state['committed_total_value'] + score
  #           alternatives.append(state)

  #       join_operation = JoinContour(i, j)
  #       new_contours = join_operation.generate_new_contour(contours)
  #       if new_contours is not None:
  #         state = dict(reference_state)
  #         state['contours_committed'] = list(reference_state['contours_committed'])
  #         state['contours'] = new_contours
  #         state['chosen_contour_index'] = i
  #         instance = expected_contours[len(state['contours_committed']) - 1]
  #         instance.prepare(new_contours[i], image_width, image_height)
  #         score = instance.shape_restrictions()
  #         state['committed_total_value'] = state['committed_total_value'] + score
  #         alternatives.append(state)

  best_alternative = None
  min_value = float('inf')
  for alternative in alternatives:
    value = alternative['committed_total_value']
    if value is not None and value < min_value:
      min_value = value
      best_alternative = alternative

  return best_alternative

def search_complete_contours(
    contours: list,
    expected_contours: list,
    search_duration_seconds: int,
    image_width: int,
    image_height: int,
    injected_start_contour: Union[np.array, None] = None,
    silent: bool = True,
    criteria_dict: dict = None
) -> list:
  if len(contours) == 0:
    return []
  
  if len(expected_contours) == 0:
    return []

  state_stack = []

  if injected_start_contour is None:
    min_score = float('inf')
    best_contour_index = -1
    for i in range(len(contours)):
      expected_contour_class = expected_contours[0]
      current_contour = contours[i]
      expected_contour_class.prepare(
        current_contour,
        image_width,
        image_height
      )
      score = expected_contour_class.shape_restrictions(criteria_dict)
      if score < min_score:
        min_score = score
        best_contour_index = i

    if min_score < float('inf'):
      state_stack.append({
        'contours_committed': [contours[best_contour_index]],
        'contours': contours,
        'chosen_contour_index': best_contour_index,
        'committed_total_value': min_score
      })
    else:
      print('No valid contours encountered (all scores are "inf"). Search stop.')
      return []
  else:
    # To be able to start the search from a specific location a start contour
    # that is meant to not be included in contours is passed as argument
    # effectively changing the first "next position restrictions" of the search.
    contours.append(injected_start_contour)
    expected_contours[0].prepare(injected_start_contour, image_width, image_height)
    score = expected_contours[0].shape_restrictions(criteria_dict)
    state_stack.append(({
      'contours_committed': [injected_start_contour],
      'contours': contours,
      'chosen_contour_index': len(contours) - 1,
      'committed_total_value': score
    }))

  start_time = time.time()
  while True:
    elapsed_time = time.time() - start_time
    if elapsed_time >= search_duration_seconds:
      return []

    if not silent:
      print(f'Elapsed time: {elapsed_time:.2f} seconds')

    if len(state_stack) > 0:
      state = state_stack[0]
      contours = state['contours']
      chosen_contour_index = state['chosen_contour_index']
      chosen_contour = contours[chosen_contour_index]
      expected_contour_class = (
        expected_contours[len(state['contours_committed']) - 1]
      )
      expected_contour_class.prepare(chosen_contour, image_width, image_height)

      if len(state['contours_committed']) == len(expected_contours):
        print('Sucessful search. Search stop.')

        return [(copy.deepcopy(
          {
            'contours_committed': state['contours_committed'],
            'committed_total_value': state['committed_total_value']
          }
        ))]
      else:
        contours_without_chosen = (
          contours[:chosen_contour_index] + contours[chosen_contour_index + 1:]
        )
        ends_branchs_sequence = False
        first_in_branch = None
        if isinstance(expected_contour_class, ExpectedContourOfBranch):
          ends_branchs_sequence = expected_contour_class.ends_branchs_sequence
          first_in_branch = expected_contour_class.first_in_branch

        contours_inside_area_indices = [
          index
          for index, contour in enumerate(contours_without_chosen)
          if is_in_allowed_space(
            contour,
            expected_contour_class,
            has_to_jump_to_next_branch=ends_branchs_sequence,
            first_in_branch=first_in_branch
          )
        ]
        if len(contours_inside_area_indices) > 0:
          best_alternative = get_best_contour_alternative(
            contours_without_chosen,
            contours_inside_area_indices,
            state,
            expected_contours,
            image_width,
            image_height,
            criteria_dict=criteria_dict
          )
          if best_alternative is not None:
            state_stack.pop(0)
            state_stack.insert(0, best_alternative)
          else:
            print('No valid contour inside required area was found for ' \
                  'expected contour ' \
                  f'index={len(state['contours_committed']) - 1}.' \
                    ' Search stop.')
            state_stack.pop(0)
        else:
          print('No contours inside required area were found for ' \
                 'expected contour ' \
                  f'index={len(state['contours_committed']) - 1}.' \
                    ' Search stop.')
          state_stack.pop(0)
      
    else:
      print('Search finished: explored all alternatives but found nothing valid.')
      return []
    
def create_minimal_image_from_contours(contours: list, padding = 0):
  if not contours:
    raise ValueError('Attempt on minimal image that covers contours done on' \
                      ' empty contours list.')

  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(np.min(x_values))
  min_y = int(np.min(y_values))
  max_x = int(np.max(x_values))
  max_y = int(np.max(y_values))

  x1 = max(0, min_x - padding)
  y1 = max(0, min_y - padding)
  x2 = max_x + padding + 1  # +1 to include the last pixel on range
  y2 = max_y + padding + 1

  return (x1, y1, x2, y2)

def measure(
    segmentation: list,
    image: np.array,
    extra_measurement_specifications: list[dict]
) -> dict:
  measurements = {}

  for segment in segmentation:
    contour = segment[0]
    instance = segment[1]
    instance.prepare(contour, image.shape[1], image.shape[0])
    local_measurements = instance.measure()
    measurements.update(local_measurements)

  for specification in extra_measurement_specifications:
    required_segments_indices = specification['required_segments_indices']
    required_segments = [
      segmentation[index] for index in required_segments_indices
    ]
    function = specification['function']
    extra_measurements = function(*required_segments)
    measurements.update(extra_measurements)

  measurement_values = measurements.values()
  min_value = min(measurement_values)
  max_value = max(measurement_values)
  measurements_normalized = {
    key: (
     (value - min_value) / (max_value - min_value)
    ) for key, value in measurements.items()
  }

  return measurements_normalized

def estimate_age(measurements: dict) -> float:
  lowest_difference = float('inf')
  age_output = None
  for age_key in BONE_AGE_ATLAS.keys():
    atlas_measurements = BONE_AGE_ATLAS[age_key]
    difference = 0
    for measurement_name in atlas_measurements.keys():
      if measurement_name in measurements:
        difference = difference + (
          np.absolute(
            atlas_measurements[measurement_name] - (
              measurements[measurement_name]
            )
          )
        )
    if difference <= lowest_difference:
      lowest_difference = difference
      age_output = age_key

  return float(age_output)

def get_expected_contours_model() -> list:
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  medial_phalanx_1 = ExpectedContourMedialPhalanx(
    encounter_amount=1,
  )
  proximal_phalanx_1 = ExpectedContourProximalPhalanx(
    encounter_amount=1,
  )
  metacarpal_1 = ExpectedContourMetacarpal(
    encounter_amount=1,
    ends_branchs_sequence=True,
    first_in_branch=distal_phalanx_1,
  )

  distal_phalanx_2 = ExpectedContourDistalPhalanx(
    encounter_amount=2,
    first_encounter=distal_phalanx_1
  )
  medial_phalanx_2 = ExpectedContourMedialPhalanx(
    encounter_amount=2,
    first_encounter=medial_phalanx_1,
  )
  proximal_phalanx_2 = ExpectedContourProximalPhalanx(
    encounter_amount=2,
    first_encounter=proximal_phalanx_1,
  )
  metacarpal_2 = ExpectedContourMetacarpal(
    encounter_amount=2,
    first_encounter=metacarpal_1,
    ends_branchs_sequence=True,
    first_in_branch=distal_phalanx_2,
  )

  distal_phalanx_3 = ExpectedContourDistalPhalanx(
    encounter_amount=3,
    first_encounter=distal_phalanx_1,
  )
  medial_phalanx_3 = ExpectedContourMedialPhalanx(
    encounter_amount=3,
    first_encounter=medial_phalanx_1,
  )
  proximal_phalanx_3 = ExpectedContourProximalPhalanx(
    encounter_amount=3,
    first_encounter=proximal_phalanx_1,
  )
  metacarpal_3 = ExpectedContourMetacarpal(
    encounter_amount=3,
    first_encounter=metacarpal_1,
    ends_branchs_sequence=True,
    first_in_branch=distal_phalanx_3
  )

  distal_phalanx_4 = ExpectedContourDistalPhalanx(
    encounter_amount=4,
    first_encounter=distal_phalanx_1,
  )
  medial_phalanx_4 = ExpectedContourMedialPhalanx(
    encounter_amount=4,
    first_encounter=medial_phalanx_1,
  )
  proximal_phalanx_4 = ExpectedContourProximalPhalanx(
    encounter_amount=4,
    first_encounter=proximal_phalanx_1,
  )
  metacarpal_4 = ExpectedContourMetacarpal(
    encounter_amount=4,
    first_encounter=metacarpal_1,
    ends_branchs_sequence=True,
    first_in_branch=distal_phalanx_4
  )

  distal_phalanx_5 = ExpectedContourDistalPhalanx(
    encounter_amount=5,
    first_encounter=distal_phalanx_1,
  )
  proximal_phalanx_5 = ExpectedContourProximalPhalanx(
    encounter_amount=5,
    first_encounter=proximal_phalanx_1,
  )
  metacarpal_5 = ExpectedContourMetacarpal(
    encounter_amount=5,
    first_encounter=metacarpal_1,
  )

  radius = ExpectedContourRadius()
  ulna = ExpectedContourUlna()

  expected_contours_model = [
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
    ulna,
    radius,
  ]
  return expected_contours_model

def estimate_age_from_image(
    input_image: Union[np.array, Image.Image],
    nofilter: bool = False,
    criteria_dict: dict = None,
) -> Tuple[float, list]:
  '''Given an image of a radiography, output the estimated age'''
  HIGHER_THRESOLD = 40
  LOWER_THRESOLD = 30
  if not nofilter:
    input_image = transforms.ToTensor()(input_image)
    he_enchanced = ContrastEnhancement().process(input_image)
    he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX,
                                cv.CV_8U)

    gaussian_blurred = cv.GaussianBlur(he_enchanced, (3, 3), 0)

    borders_detected = cv.Canny(gaussian_blurred, HIGHER_THRESOLD, 135)
    borders_detected = cv.normalize(borders_detected, None, 0, 255,
                                    cv.NORM_MINMAX, cv.CV_8U)
    
    borders_detected_2 = cv.Canny(gaussian_blurred, LOWER_THRESOLD, 135)
    borders_detected_2 = cv.normalize(borders_detected_2, None, 0, 255,
                                    cv.NORM_MINMAX, cv.CV_8U)
  else:
    # Using the same thresold even tho the image may not be a canny filter
    # result.

    borders_detected = np.array(input_image)
    borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(borders_detected, HIGHER_THRESOLD, 255,
                             cv.THRESH_BINARY)
    borders_detected = thresh
    borders_detected_2 = thresh

  # Segmentation
  contours_2, _ = cv.findContours(borders_detected_2, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
  contours_2 = list(contours_2)

  (
    x1,
    y1,
    x2,
    y2
  ) = create_minimal_image_from_contours(contours_2)

  PADDING = 20
  minimum_image_1 = borders_detected[y1:y2, x1:x2]
  minimum_image_1 = cv.copyMakeBorder(
    minimum_image_1,
    PADDING,
    PADDING,
    PADDING,
    PADDING,
    cv.BORDER_CONSTANT,
    value=(0, 0, 0)
  )
  minimum_image_2 = borders_detected_2[y1:y2, x1:x2]
  minimum_image_2 = cv.copyMakeBorder(
    minimum_image_2,
    PADDING,
    PADDING,
    PADDING,
    PADDING,
    cv.BORDER_CONSTANT,
    value=(0, 0, 0)
  )

  contours, _ = cv.findContours(minimum_image_1, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)
  contours = list(contours)
  contours_2, _ = cv.findContours(minimum_image_2, cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)
  contours_2 = list(contours_2)

  expected_contours = get_expected_contours_model()
  complete_contours = search_complete_contours(
    contours,
    expected_contours,
    SEARCH_EXECUTION_DURATION_SECONDS,
    image_width=minimum_image_1.shape[1],
    image_height=minimum_image_1.shape[0],
    criteria_dict=criteria_dict
  )

  if len(complete_contours) > 0:
    best_contours = complete_contours[0]['contours_committed']

    metacarpal2 = best_contours[7]
    metacarpal3 = best_contours[11]
    metacarpal4 = best_contours[15]
    metacarpal5 = best_contours[18]
    ulna = best_contours[19]
    radius = best_contours[20]
    segmentation = [
      [metacarpal2, expected_contours[7]],
      [metacarpal3, expected_contours[11]],
      [metacarpal4, expected_contours[15]],
      [metacarpal5, expected_contours[18]],
      [ulna, expected_contours[19]],
      [radius, expected_contours[20]],
    ]

    # Second search to find the sesamoid
    expected_contours_2 = [
      ExpectedContourSesamoidMetacarpal(),
      ExpectedContourSesamoid(),
    ]
    complete_contours_2 = search_complete_contours(
      contours_2,
      expected_contours_2,
      SEARCH_EXECUTION_DURATION_SECONDS,
      image_width=minimum_image_2.shape[1],
      image_height=minimum_image_2.shape[0],
      injected_start_contour=metacarpal5,
      criteria_dict=criteria_dict,
    )

    if len(complete_contours_2) > 0:
      best_contours_2 = complete_contours_2[0]['contours_committed']

      sesamoid = best_contours_2[1]
      segmentation = [
        *segmentation,
        [sesamoid, expected_contours_2[1]]
      ]
      
      def intersesamoid_distance_measurement(
          metacarpal_segment,
          sesamoid_segment
      ) -> dict:
        metacarpal_contour = metacarpal_segment[0]
        sesamoid_contour = sesamoid_segment[0]

        min_distance = float('inf')

        for metacarpal_point in metacarpal_contour:
          for sesamoid_point in sesamoid_contour:
            distance = np.sqrt(np.sum((metacarpal_point - sesamoid_point) ** 2))
            if distance < min_distance:
              min_distance = distance
        
        return { "inter-sesamoid_distance": float(min_distance) }

      extra_measurement_specification = {
        'required_segments_indices': [3, 6],
        'function': intersesamoid_distance_measurement
      }

      # minimum_image_2 will always be bigger than 1 because it has a lower
      # threshold so at least as many contours as 1 will be detected.
      measurements_normalized = measure(
        segmentation,
        minimum_image_2,
        [extra_measurement_specification]
      )

      estimated_age = estimate_age(measurements_normalized)

      return (
        estimated_age,
        measurements_normalized,
        minimum_image_1,
        minimum_image_2,
      )
    else:
      return (
        -1,
        None,
        minimum_image_1,
        minimum_image_2,
      )
  else:
    return (
      -2,
      None,
      minimum_image_1,
      minimum_image_2,
    )

def process_radiograph(
    filename: str,
    write_images: bool = False,
    show_images: bool = False,
    nofilter: bool = False,
    all: bool = False
) -> None:
  input_image = None
  try:
    with Image.open(filename) as image:
      input_image = np.array(image)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  (
    estimated_age,
    measurements,
    image_search_stage_1,
    image_search_stage_2,
  ) = estimate_age_from_image(input_image, nofilter)


  if not all:
    if estimated_age == -1:
      print('Could not estimate age due to unsuccessful second search' \
        ' for sesamoid segmentation.')
    elif estimated_age == -2:
      print('Could not estimate age due to unsuccessful bone segmentation.')
    elif estimated_age < 18:
      print('System finds patient age < 18.')
      print('Patient is underage.')
    else:
      print('System finds patient age => 18.')
      print('Patient is adult.')
  else:
    if estimated_age == -1:
      print('Could not estimate age due to unsuccessful second search' \
        ' for sesamoid segmentation.')
    elif estimated_age == -2:
      print('Could not estimate age due to unsuccessful bone segmentation.')
    elif estimated_age < 18:
      print('System finds patient age < 18.')
      print('Patient is underage.\n')
      print(f'Estimated age is {estimated_age}.\n')
      print('Measurements:')
      for measurement_key in measurements:
        measurement_value = measurements[measurement_key]
        print(f'{measurement_key}={measurement_value}')
      print('\n')
    else:
      print('System finds patient age => 18.')
      print('Patient is adult.\n')
      print(f'Estimated age is {estimated_age}.\n')
      print('Measurements:')
      for measurement_key in measurements:
        measurement_value = measurements[measurement_key]
        print(f'{measurement_key}={measurement_value}')
      print('\n')

  if write_images:
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
               f'execute_output_stage_1.jpg',
               image_search_stage_1)
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
               f'execute_output_stage_2.jpg',
               image_search_stage_2)

  if show_images:
    fig = plt.figure()
    plt.imshow(image_search_stage_1)
    plt.title('Image output of search stage 1')
    plt.axis('off')
    fig.canvas.manager.set_window_title('Minimum image 1')

    fig = plt.figure()
    plt.imshow(image_search_stage_2)
    plt.title('Image output of search stage 2')
    plt.axis('off')
    fig.canvas.manager.set_window_title('Minimum image 2')
    plt.show()
