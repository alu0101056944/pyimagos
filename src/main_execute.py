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
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.metacarpal_sesamoid import ExpectedContourMetacarpalSesamoid
from constants import EXECUTION_DURATION_SECONDS
from src.contour_operations.cut_contour import CutContour
from src.contour_operations.extend_contour import ExtendContour
from src.contour_operations.join_contour import JoinContour
from constants import BONE_AGE_ATLAS

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
  first_encounter=distal_phalanx_1
)
medial_phalanx_2 = ExpectedContourMedialPhalanx(
  encounter_amount=2,
  first_encounter=medial_phalanx_1,
  first_in_branch=distal_phalanx_2
)
proximal_phalanx_2 = ExpectedContourProximalPhalanx(
  encounter_amount=2,
  first_encounter=proximal_phalanx_1,
  first_in_branch=distal_phalanx_2
)
metacarpal_2 = ExpectedContourMetacarpal(
  encounter_amount=2,
  first_encounter=metacarpal_1,
  first_in_branch=distal_phalanx_2
)

distal_phalanx_3 = ExpectedContourDistalPhalanx(
  encounter_amount=3,
  first_encounter=distal_phalanx_1,
)
medial_phalanx_3 = ExpectedContourMedialPhalanx(
  encounter_amount=3,
  first_encounter=medial_phalanx_1,
  first_in_branch=distal_phalanx_3
)
proximal_phalanx_3 = ExpectedContourProximalPhalanx(
  encounter_amount=3,
  first_encounter=proximal_phalanx_1,
  first_in_branch=distal_phalanx_3
)
metacarpal_3 = ExpectedContourMetacarpal(
  encounter_amount=3,
  first_encounter=metacarpal_1,
  first_in_branch=distal_phalanx_3
)

distal_phalanx_4 = ExpectedContourDistalPhalanx(
  encounter_amount=4,
  first_encounter=distal_phalanx_1,
)
medial_phalanx_4 = ExpectedContourMedialPhalanx(
  encounter_amount=4,
  first_encounter=medial_phalanx_1,
  first_in_branch=distal_phalanx_4
)
proximal_phalanx_4 = ExpectedContourProximalPhalanx(
  encounter_amount=4,
  first_encounter=proximal_phalanx_1,
  first_in_branch=distal_phalanx_4
)
metacarpal_4 = ExpectedContourMetacarpal(
  encounter_amount=4,
  first_encounter=metacarpal_1,
  first_in_branch=distal_phalanx_4
)

distal_phalanx_5 = ExpectedContourDistalPhalanx(
  encounter_amount=5,
  first_encounter=distal_phalanx_1,
)
proximal_phalanx_5 = ExpectedContourProximalPhalanx(
  encounter_amount=5,
  first_encounter=proximal_phalanx_1,
  first_in_branch=distal_phalanx_5
)
metacarpal_5 = ExpectedContourMetacarpal(
  encounter_amount=5,
  first_encounter=metacarpal_1,
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
  ulna,
  radius,
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
          print('Found an instance of contours that covers all expected' \
                ' contours.')

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
          state,
          image_width,
          image_height
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

def create_minimal_image_from_contours(image: np.array,
                                       contours: list,
                                       padding = 0) -> np.array:
  if not contours:
    raise ValueError('Called main_execute.py:' \
                     'create_minimal_image_from_contours(<contour>) with an ' \
                      'empty contours array')
  
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  min_x = int(max(0, np.min(x_values)))
  min_y = int(max(0, np.min(y_values)))
  max_x = int(min(image.shape[1], np.max(x_values)))
  max_y = int(min(image.shape[0], np.max(y_values)))

  roi_from_original = image[
    max(0, min_y - padding):max_y + padding + 1,
    max(0, min_x - padding):max_x + padding + 1
  ]
  roi_from_original = np.copy(roi_from_original)

  # missing X padding correction on the left
  if np.min(x_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(x_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=1,
      dtype=np.uint8
    )
  
  # missing X padding correction on the right
  if np.max(x_values) + padding > image.shape[1]:
    missing_pixel_amount = np.max(x_values) + padding - image.shape[1]
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((roi_from_original.shape[0], missing_pixel_amount), 0,
                dtype=np.uint8)
      ),
      axis=1,
      dtype=np.uint8
    )

  # missing Y padding correction on top
  if np.min(y_values) - padding < 0:
    missing_pixel_amount = np.absolute(np.min(y_values) - padding)
    roi_from_original = np.concatenate(
      (
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
        roi_from_original,
      ),
      axis=0,
      dtype=np.uint8
    )
  
  # missing Y padding correction on bottom
  if np.max(y_values) + padding > image.shape[0]:
    missing_pixel_amount = np.max(y_values) + padding - image.shape[0]
    roi_from_original = np.concatenate(
      (
        roi_from_original,
        np.full((missing_pixel_amount, roi_from_original.shape[1]), 0,
                dtype=np.uint8),
      ),
      axis=0,
      dtype=np.uint8
    ) 

  corrected_contours = [
    points - np.array([[[min_x, min_y]]]) + padding for points in contours
  ]

  return roi_from_original, corrected_contours

def get_bounding_box_xy(contour):
  x, y, w, h = cv.boundingRect(contour)
  return x, y

def create_minimal_image_from_contour(image: np.array,
                                       contour: np.array,
                                       padding: int) -> np.array:
  x, y, w, h = cv.boundingRect(contour)

  x1 = max(0, x - padding)
  y1 = max(0, y - padding)
  x2 = min(image.shape[1], x + w + padding)
  y2 = min(image.shape[0], y + h + padding)

  cropped_image = image[y1:y2, x1:x2]

  return cropped_image

def get_contours_in_region(image, contours, region_rect, padding):
  x, y, w, h = region_rect
  region_x1 = max(0, x - padding)
  region_y1 = max(0, y - padding)
  region_x2 = min(image.shape[1], x + w + padding)
  region_y2 = min(image.shape[0], y + h + padding)

  contours_in_region = []

  for contour in contours:
    contour_x, contour_y, contour_w, contour_h = cv.boundingRect(contour)
    contour_x1 = contour_x
    contour_y1 = contour_y
    contour_x2 = contour_x + contour_w
    contour_y2 = contour_y + contour_h

    if (
      region_x1 <= contour_x1
      and region_y1 <= contour_y1
      and region_x2 >= contour_x2
      and region_y2 >= contour_y2
    ):
      contours_in_region.append(contour)

  return contours_in_region

def find_sesamoid(image: np.array, segmentation: dict, contours: list) -> list:
  focused_image = create_minimal_image_from_contour(
    image,
    segmentation['metacarpal5'],
    padding=10
  )

  region_rect = cv.boundingRect(segmentation['metacarpal5'])
  contours_within_focused_image = get_contours_in_region(
    focused_image,
    contours,
    region_rect, #TODO change rect region to image and not contour
    padding=10
  )

  shape_values = []
  sesamoid_instance = ExpectedContourMetacarpalSesamoid()
  for contour_within in contours_within_focused_image:
    sesamoid_instance.prepare(
      contour_within,
      focused_image.shape[1],
      focused_image.shape[0]
    )
    shape_value = sesamoid_instance.shape_restrictions()
    shape_values.append(shape_value)

  if len(shape_values) > 0:
    sesamoid_contour_index = np.argmax(shape_values)
    sesamoid_contour = contours[sesamoid_contour_index]
  else:
    sesamoid_contour = None

  return sesamoid_contour

def _find_closest_pair(contour_a: list, contour_b: list):
  overall_min_distance = float('inf')
  overall_closest_pair = None

  for i, point_a in enumerate(contour_a):
    distances = np.sqrt(np.sum((contour_b - point_a) ** 2, axis=1))

    sorted_indices = np.argsort(distances)

    min_distance_local = distances[sorted_indices[0]]
    min_distance_index = sorted_indices[0]
    second_min_distance_index = sorted_indices[1] if len(distances) > 1 else None

    if min_distance_local < overall_min_distance:
      overall_min_distance = min_distance_local
      overall_closest_pair = (i, min_distance_index, second_min_distance_index)
  
  if overall_closest_pair is not None:
    index_a = overall_closest_pair[0]
    min_index = overall_closest_pair[1]
    second_min_index = overall_closest_pair[2]
    return index_a, min_index, second_min_index
  else:
    return None
  
def measure(segmentation, image: np.array, contours: list) -> dict:
  measurements = {}

  metacarpal_instances=[
    metacarpal_2,
    metacarpal_3,
    metacarpal_4,
    metacarpal_5,
  ]
  for i in range(len(metacarpal_instances)):
    instance = metacarpal_instances[i]
    instance.prepare(
      segmentation[f'metacarpal{i + 2}'],
      image_width=image.shape[1],
      image_height=image.shape[0]
    )
    measurements_dict = instance.measure()
    measurements.update(measurements_dict)

  instance = ulna
  instance.prepare(
    segmentation['ulna'],
    image_width=image.shape[1],
    image_height=image.shape[0]
  )
  measurements_dict = instance.measure()
  measurements.update(measurements_dict)

  instance = radius
  instance.prepare(
    segmentation['radius'],
    image_width=image.shape[1],
    image_height=image.shape[0]
  )
  measurements_dict = instance.measure()
  measurements.update(measurements_dict)

  # sesamoid measurements
  sesamoid_contour = find_sesamoid(
    image,
    segmentation,
    contours
  )
  if sesamoid_contour:
    index_a, closest_index_b, _ = _find_closest_pair(
      segmentation['metacarpal5'],
      sesamoid_contour
    )

    sesamoid_distance = np.sqrt(
      np.sum(
        (
          segmentation['metacarpal5'][index_a] - (
          sesamoid_contour[closest_index_b])
        ) ** 2
      )
    )

    measurements['inter-sesamoid_distance'] = sesamoid_distance
  
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

def process_radiograph(filename: str,
                       write_images: bool = False,
                       show_images: bool = False,
                       nofilter: bool = False,
                       nosearch: bool = False,
                       historygui: bool = False
                      ) -> None:
  global expected_contours
  if len(expected_contours) == 0:
     raise ValueError("The global list 'expected_contours' must have at least one' \
                      ' element to work correctly.")

  input_image = Image.open(filename)

  if not nofilter:
    input_image = transforms.ToTensor()(input_image)
    he_enchanced = ContrastEnhancement().process(input_image)
    he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX,
                                cv.CV_8U)

    gaussian_blurred = cv.GaussianBlur(he_enchanced, (5, 5), 0)
    borders_detected = cv.Canny(gaussian_blurred, 40, 135)
    borders_detected = cv.normalize(borders_detected, None, 0, 255, cv.NORM_MINMAX,
                                    cv.CV_8U)
  else:
    borders_detected = np.array(input_image)
    borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
    borders_detected = thresh

  # Segmentation
  successful_segmentation = False

  contours, _ = cv.findContours(borders_detected, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
  
  minimum_image, corrected_contours = create_minimal_image_from_contours(
    borders_detected,
    contours
  )
  contours = corrected_contours
  image_width = minimum_image.shape[1]
  image_height = minimum_image.shape[0]

  if not nosearch:
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
                                                EXECUTION_DURATION_SECONDS,
                                                image_width,
                                                image_height)
    complete_contours.sort(key=lambda item: item[1], reverse=True)
    if len(complete_contours) > 0:
      successful_segmentation = True
      best_contours = complete_contours[0]

    if historygui:
      # Opens history gui
      _ = ContourViewer(minimum_image, complete_contours)
  else:
    best_contours = contours

  if successful_segmentation:
    segmentation = {
      'metacarpal2': best_contours[7],
      'metacarpal3': best_contours[11],
      'metacarpal4': best_contours[15],
      'metacarpal5': best_contours[18],
      'ulna': best_contours[19],
      'radius': best_contours[20]
    }
  
    measurements_normalized = measure(segmentation, minimum_image.shape[1],
                                      minimum_image.shape[0])

    estimated_age = estimate_age(measurements_normalized)

    if estimated_age < 18:
      print('System finds patient age <18.')
      print('Patient is underage.')
    else:
      print('System finds patient age =>18.')
      print('Patient is adult.')
  else:
    print('Could not estimate age due to unsuccessful bone segmentation.')
  if write_images:
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
               f'execute_output.jpg',
               minimum_image)
  else:
    if show_images:
      cv.imshow(f'{os.path.basename(filename)}' \
                f'execute_output.jpg',
                minimum_image)
      cv.waitKey(0)
      cv.destroyAllWindows()
