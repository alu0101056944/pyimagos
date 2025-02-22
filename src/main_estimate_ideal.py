'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Given a perfect borders detected image, segment contours and classify
each contour. Then measure and estimate.
'''

from PIL import Image

import numpy as np
import cv2 as cv

from src.main_develop_contours_gui import (
  draw_contours, draw_contour_points
)
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.metacarpal_sesamoid import (
  ExpectedContourMetacarpalSesamoid
)
from constants import BONE_AGE_ATLAS

def get_bounding_box_xy(contour):
  x, y, w, h = cv.boundingRect(contour)
  return x, y


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

def find_sesamoid(image: np.array, contours: list) -> list:
  sorted_contours = sorted(contours, key=get_bounding_box_xy)
  focused_image = create_minimal_image_from_contour(
    image,
    sorted_contours[0],
    padding=10
  )

  region_rect = cv.boundingRect(sorted_contours[0])
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

def case_1():
  borders_detected = Image.open('docs/radiography_ideal_a_with_sesamoid_minimal.jpg')
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

  sorted_contours = sorted(contours, key=get_bounding_box_xy)
  segmentation = {
    'metacarpal1': sorted_contours[1],
    'metacarpal2': sorted_contours[3],
    'metacarpal3': sorted_contours[5],
    'metacarpal4': sorted_contours[6],
    'metacarpal5': sorted_contours[7],
    'intersesamoid': sorted_contours[8],
    'ulna': sorted_contours[0],
    'radius': sorted_contours[4]
  }

  # Used to map the contours by consecutive executions:
  # for point in segmentation['intersesamoid']:
  #   x, y = point[0]
  #   x = x + 2
  #   contours_drawn_image[y, x] = (0, 0, 255)

  # cv.imshow('Drawn contours', contours_drawn_image)
  # cv.waitKey(0)
  # cv.destroyAllWindows()

  METACARPAL_BONES_AMOUNT = 5
  measurements = {}
  for i in range(METACARPAL_BONES_AMOUNT):
    instance = ExpectedContourMetacarpal(i + 1)
    instance.prepare(
      segmentation[f'metacarpal{i + 1}'],
      image_width=contours_drawn_image.shape[1],
      image_height=contours_drawn_image.shape[0]
    )
    measurements_dict = instance.measure()
    measurements.update(measurements_dict)

  instance = ExpectedContourUlna()
  instance.prepare(
    segmentation['ulna'],
    image_width=contours_drawn_image.shape[1],
    image_height=contours_drawn_image.shape[0]
  )
  measurements_dict = instance.measure()
  measurements.update(measurements_dict)

  instance = ExpectedContourRadius()
  instance.prepare(
    segmentation['radius'],
    image_width=contours_drawn_image.shape[1],
    image_height=contours_drawn_image.shape[0]
  )
  measurements_dict = instance.measure()
  measurements.update(measurements_dict)


  # get the sesamoid contour and then its measurements
  sesamoid_contour = find_sesamoid(contours_drawn_image, contours)
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

  # Compare measurements with measurements atlas
  lowest_difference = float('inf')
  age_output = None
  for age_key in BONE_AGE_ATLAS.keys():
    atlas_measurements = BONE_AGE_ATLAS[age_key]
    difference = 0
    for measurement_name in atlas_measurements.keys():
      if measurement_name in measurements_normalized:
        difference = difference + (
          np.absolute(
            atlas_measurements[measurement_name] - (
              measurements_normalized[measurement_name]
            )
          )
        )
    if difference <= lowest_difference:
      lowest_difference = difference
      age_output = age_key

  if float(age_output) < 18:
    print('Sistema encuentra edad de individuo <18.')
    print('El individuo es menor de edad.')
  else:
    print('Sistema encuentra edad de individuo >18.')
    print('El individuo es mayor de edad.')

def estimate_age_from_ideal_contour():
  case_1()
