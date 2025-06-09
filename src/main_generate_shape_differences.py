'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate a table showing all the shape failures on the manual segmentation.
'''

import numpy as np

from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.proximal_phalanx import ExpectedContourProximalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.radius import ExpectedContourRadius

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
from src.radiographies.rad_013 import case_013
from src.radiographies.rad_016 import case_016
from src.radiographies.rad_019 import case_019
from src.radiographies.rad_030 import case_030
from src.radiographies.rad_031 import case_031
from src.radiographies.rad_084 import case_084
from src.radiographies.rad_1619 import case_1619
from src.radiographies.rad_1779 import case_1779
from src.radiographies.rad_2089 import case_2089

from src.main_develop_test_distal_phalanx import (
  create_minimal_image_from_contours,
)

def minimize_contours(contours):
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
    padding=20,
  )
  return adjusted_contours

def contour_shape_failures(contours: list, contour_map: list,
                      expected_contours: list, title: str) -> list[str]:
  '''contour_map is the ordered version of contours which corresponds to the
  expected_contours sequence, which is a list of ExpectedContour classes.'''
  own_title_to_contour_acceptance = {title: []}

  for i in range(len(contour_map)):
    contour = contours[contour_map[i]]
    expected_contour = expected_contours[i]
    
    points = np.reshape(contour, (-1, 2))
    if len(points) == 0:
      output_string = output_string + (
        f'Contour {i} is empty. Skipping.\n')
    
    all_x = points[:, 0]
    all_y = points[:, 1]
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)
    image_width = int(max_x - min_x)
    image_height = int(max_y - min_y)
    expected_contour.prepare(contour, image_width, image_height)
    score = expected_contour.shape_restrictions()
    if score != float('inf'):
      own_title_to_contour_acceptance[title].append(True)
    else:
      own_title_to_contour_acceptance[title].append(False)

  return own_title_to_contour_acceptance


def case_004_local():
  contours = case_004()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 9
  metacarpal_1 = 4
  distal_2 = 19
  medial_2 = 15
  proximal_2 = 10
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '004 radiography'
  )


def case_022_local():
  contours = case_022()
  
  distal_1 = 13
  medial_1 = 10
  proximal_1 = 8
  metacarpal_1 = 3
  distal_2 = 18
  medial_2 = 15
  proximal_2 = 11
  metacarpal_2 = 4
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 14
  metacarpal_3 = 7
  distal_4 = 19
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 9
  proximal_5 = 5
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '022 radiography'
  )

def case_006_local():
  contours = case_006()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 8
  metacarpal_1 = 3
  distal_2 = 18
  medial_2 = 15
  proximal_2 = 10
  metacarpal_2 = 4
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 6
  distal_4 = 19
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 7
  distal_5 = 9
  proximal_5 = 5
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '006 radiography'
  )

def case_018_local():
  contours = case_018()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 9
  metacarpal_1 = 4
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 12
  metacarpal_2 = 6
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 11
  metacarpal_4 = 7
  distal_5 = 5
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '018 radiography'
  )

def case_023_local():
  contours = case_023()
  distal_1 = 14
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 4
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 18
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 17
  medial_4 = 15
  proximal_4 = 10
  metacarpal_4 = 7
  distal_5 = 6
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '023 radiography'
  )

def case_029_local():
  contours = case_029()
  distal_1 = 14
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 5
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 6
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 10
  metacarpal_4 = 7
  distal_5 = 4
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '029 radiography'
  )

def case_032_local():
  contours = case_032()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 8
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 15
  proximal_2 = 11
  metacarpal_2 = 4
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 6
  distal_4 = 18
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 7
  distal_5 = 9
  proximal_5 = 5
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '032 radiography'
  )

def case_217_local():
  contours = case_217()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 18
  medial_2 = 15
  proximal_2 = 10
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 19
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 8
  distal_5 = 6
  proximal_5 = 4
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '217 radiography'
  )

def case_1622_local():
  contours = case_1622()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 8
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 4
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 9
  proximal_5 = 5
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '1622 radiography'
  )

def case_1886_local():
  contours = case_1886()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 12
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 11
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 4
  metacarpal_5 = 2
  ulna = 0
  radio = 1

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '1886 radiography'
  )

def case_013_local():
  contours = case_013()
  distal_1 = 14
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 10
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 11
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 4
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '013 radiography'
  )

def case_016_local():
  contours = case_016()
  distal_1 = 14
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 4 
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 10
  metacarpal_4 = 7
  distal_5 = 6
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '016 radiography'
  )

def case_019_local():
  contours = case_019()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 4
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '019 radiography'
  )

def case_030_local():
  contours = case_030()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 18
  medial_2 = 15
  proximal_2 = 10
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 19
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 4
  metacarpal_5 = 2 
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '030 radiography'
  )

def case_031_local():
  contours = case_031()
  distal_1 = 14
  medial_1 = 10
  proximal_1 = 9
  metacarpal_1 = 5
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 6
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 12
  metacarpal_4 = 7
  distal_5 = 4
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '031 radiography'
  )

def case_084_local():
  contours = case_084()
  distal_1 = 15
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 5
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 8
  distal_3 = 20
  medial_3 = 18
  proximal_3 = 13
  metacarpal_3 = 2
  distal_4 = 17
  medial_4 = 14
  proximal_4 = 10
  metacarpal_4 = 7
  distal_5 = 6
  proximal_5 = 4 
  metacarpal_5 = 3
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '084 radiography'
  )

def case_1619_local():
  contours = case_1619()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 15
  proximal_2 = 10
  metacarpal_2 =4
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 16
  proximal_4 = 12
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 5
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '1619 radiography'
  )

def case_1779_local():
  contours = case_1779()
  distal_1 = 14
  medial_1 = 11
  proximal_1 = 9
  metacarpal_1 = 4
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 12
  metacarpal_2 = 7
  distal_3 = 20
  medial_3 = 18
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 17
  medial_4 = 15
  proximal_4 = 10
  metacarpal_4 = 6
  distal_5 = 5
  proximal_5 = 3
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '1779 radiography'
  )

def case_2089_local():
  contours = case_2089()
  distal_1 = 14
  medial_1 = 12
  proximal_1 = 9
  metacarpal_1 = 3
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 5
  distal_3 = 20
  medial_3 = 17
  proximal_3 = 13
  metacarpal_3 = 7
  distal_4 = 18
  medial_4 = 15
  proximal_4 = 10
  metacarpal_4 = 6
  distal_5 = 8
  proximal_5 = 4
  metacarpal_5 = 2
  ulna = 1
  radio = 0

  contour_map = [
    distal_1, medial_1, proximal_1, metacarpal_1,
    distal_2, medial_2, proximal_2, metacarpal_2,
    distal_3, medial_3, proximal_3, metacarpal_3,
    distal_4, medial_4, proximal_4, metacarpal_4,
    distal_5, proximal_5, metacarpal_5,
    ulna,
    radio,
  ]

  expected_contours = get_canonical_expected_contours()

  return contour_shape_failures(
    contours,
    contour_map,
    expected_contours,
    '2089 radiography'
  )

def get_canonical_expected_contours():
  '''Does not include sesamoid'''
  expected_contours = np.empty(21, dtype=object)
  expected_contours[0] = ExpectedContourDistalPhalanx(1)
  expected_contours[1] = ExpectedContourMedialPhalanx(1)
  expected_contours[2] = ExpectedContourProximalPhalanx(1)
  expected_contours[3] = ExpectedContourMetacarpal(
    1,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[0]
  )
  expected_contours[4] = ExpectedContourDistalPhalanx(
    2,
    previous_encounter=expected_contours[0]
  )
  expected_contours[5] = ExpectedContourMedialPhalanx(
    2,
    previous_encounter=expected_contours[1]
  )
  expected_contours[6] = ExpectedContourProximalPhalanx(
    2,
    previous_encounter=expected_contours[2]
  )
  expected_contours[7] = ExpectedContourMetacarpal(
    2,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[4],
    previous_encounter=expected_contours[3]
  )
  expected_contours[8] = ExpectedContourDistalPhalanx(
    3,
    previous_encounter=expected_contours[0]
  )
  expected_contours[9] = ExpectedContourMedialPhalanx(
    3,
    previous_encounter=expected_contours[1]
  )
  expected_contours[10] = ExpectedContourProximalPhalanx(
    3,
    previous_encounter=expected_contours[2]
  )
  expected_contours[11] = ExpectedContourMetacarpal(
    3,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[8],
    previous_encounter=expected_contours[3]
  )
  expected_contours[12] = ExpectedContourDistalPhalanx(
    4,
    previous_encounter=expected_contours[0]
  )
  expected_contours[13] = ExpectedContourMedialPhalanx(
    4,
    previous_encounter=expected_contours[1]
  )
  expected_contours[14] = ExpectedContourProximalPhalanx(
    4,
    previous_encounter=expected_contours[2]
  )
  expected_contours[15] = ExpectedContourMetacarpal(
    4,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[12],
    previous_encounter=expected_contours[3]
  )
  expected_contours[16] = ExpectedContourDistalPhalanx(
    5,
    previous_encounter=expected_contours[0]
  )
  expected_contours[17] = ExpectedContourProximalPhalanx(
    5,
    previous_encounter=expected_contours[2]
  )
  expected_contours[18] = ExpectedContourMetacarpal(
    5,
    previous_encounter=expected_contours[3]
  )
  expected_contours[19] = ExpectedContourUlna()
  expected_contours[20] = ExpectedContourRadius()

  return expected_contours

def generate_shape_differences_table_main():
  output_string = ''

  case_to_contour_acceptances = {}
  case_to_contour_acceptances.update(case_004_local())
  case_to_contour_acceptances.update(case_022_local())
  case_to_contour_acceptances.update(case_006_local())
  case_to_contour_acceptances.update(case_018_local())
  case_to_contour_acceptances.update(case_023_local())
  case_to_contour_acceptances.update(case_029_local())
  case_to_contour_acceptances.update(case_032_local())
  case_to_contour_acceptances.update(case_217_local())
  case_to_contour_acceptances.update(case_1622_local())
  case_to_contour_acceptances.update(case_1886_local())
  case_to_contour_acceptances.update(case_013_local())
  case_to_contour_acceptances.update(case_016_local())
  case_to_contour_acceptances.update(case_019_local())
  case_to_contour_acceptances.update(case_030_local())
  case_to_contour_acceptances.update(case_031_local())
  case_to_contour_acceptances.update(case_084_local())
  case_to_contour_acceptances.update(case_1619_local())
  case_to_contour_acceptances.update(case_1779_local())
  case_to_contour_acceptances.update(case_2089_local())

  def table_start(encounter_amount: int = 1):
    col_specs_str = ''.join(['>{\\raggedright\\tiny\\arraybackslash}p{0.005\\textwidth}|'] * 22)
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Validez de la forma de los contornos respecto a segmentación manual ') + (
        f'{encounter_amount}' + '}\n') + (
      '\\label{tab:shape_differences' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|' + col_specs_str +'}\n') + (
      '\\toprule\n') + (
      ' & \\multicolumn{21}{c|}{Índices de contornos} \\\\ \n') + (
      '\\hline\n \\textbf{Archivo} & ') + (
          ' & '.join([str(value) for value in list(np.arange(0, 21, 1))])
        ) + ' & V ' ' \\\\ \n' + ('\\hline\n')
  
  def table_end():
    return (
      '\\bottomrule\n') + (
      '\\end{tabular}\n') + (
      '\\end{footnotesize}\n') + (
      '\\end{table}\n')
  
  def cut_table(new_line: str, encounter_amount):
    return table_end() + '\n' + table_start(encounter_amount) + new_line

  output_string = output_string + table_start()
  
  tables_generated = 1
  table_cut_count = 0
  TABLE_CUT_THRESHOLD = 12

  for case_title in case_to_contour_acceptances:
    processed_title = case_title.replace('_', '\\_')
    contour_acceptances = [
      '1' if value == True else '0' for value in case_to_contour_acceptances[case_title]
    ]
    case_valid = contour_acceptances.count('1') == len(contour_acceptances)
    table_entry = f'{processed_title} & {' & '.join(contour_acceptances)} ' + (
      f'& {'1' if case_valid else '0'} \\\\ \n')

    table_cut_count += 1
    if table_cut_count >= TABLE_CUT_THRESHOLD:
      tables_generated += 1
      output_string = output_string + cut_table(table_entry,
                                                encounter_amount=tables_generated)
      table_cut_count = 0
    else:
      output_string = output_string + table_entry
    output_string = output_string + '\\hline\n'

  output_string = output_string + table_end()

  with open('tab_shape_differences.txt', 'w', encoding='utf-8') as f:
    f.write(output_string)
    print('Writing tab_shape_differences.txt')
    print('Success.')
