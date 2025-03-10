'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Print per contour-position_restriction the maximum difference between the restriction
line's y and a contour point's y (only on those that are on the wrong side)

'''

import numpy as np

from src.expected_contours.distal_phalanx import (
  AllowedLineSideBasedOnYorXOnVertical
)
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

from src.main_experiment_positions import count_invasion_factor_max

def count_invasion_factor(contour: np.array,
                          position_restrictions: list) -> list[float]:
  '''The invasion factor is bigger the more of the contour is in the wrong
  area.'''

  invasion_factors = []
  for position_restriction in position_restrictions:
    p1, p2, allowed_side_array = position_restriction
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    local_invasion_factor = 0
    if x2 == x1:
      allowed_side = allowed_side_array[3]
      x_line = x1

      for point in contour:
        x = point[0][0]
        if allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER:
          if x <= x_line:
            local_invasion_factor += (x_line - x)
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
          if x >= x_line:
            local_invasion_factor += (x - x_line)
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
          if x < x_line:
            local_invasion_factor += (x_line - x)
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
          if x > x_line:
            local_invasion_factor += (x - x_line)
        else:
          m = (y2 - y1) / (x2 - x1)
          b = y1 - m * x1
          if m > 0:
            allowed_side = allowed_side_array[0]
          elif m < 0:
            allowed_side = allowed_side_array[1]
          else:
            allowed_side = allowed_side_array[2]
          
          for point in contour:
            x = point[0][0]
            y_point = point[0][1]
            line_y = m * x + b
            if allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER:
              if y_point <= line_y:
                local_invasion_factor += (line_y - y_point)
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
              if y_point >= line_y:
                local_invasion_factor += (y_point - line_y)
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
              if y_point < line_y:
                local_invasion_factor += (line_y - y_point)
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
              if y_point > line_y:
                local_invasion_factor += (y_point - line_y)
    
    invasion_factors.append(local_invasion_factor)

  return invasion_factors

def get_string_differences(contours: list, contour_map: list,
                      expected_contours: list, title: str) -> list[str]:
  '''contour_map is the ordered version of contours which corresponds to the
  expected_contours sequence, which is a list of ExpectedContour classes.'''
  output_string = f'Printing invasion factors for {title}\n'
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
    
    position_restrictions = expected_contour.next_contour_restrictions()
    invasion_factors = count_invasion_factor_max(contour, position_restrictions)


    for j, invasion_factor in enumerate(invasion_factors):
      output_string = output_string + (
        f'Contour {i}, factor {j} (type={type(expected_contour).__name__}' \
        f'): invasion factor={invasion_factor}\n')
      
    if len(invasion_factors) > 0:
      local_max_invasion_factors = max(invasion_factors)
      output_string = output_string + (
        f'Contour {i} (type={type(expected_contour).__name__}): ' \
        f'local max invasion factor={local_max_invasion_factors}\n')
    else:
      output_string = output_string + (
        f'Contour {i} (type={type(expected_contour).__name__}): ' \
        f'local total invasion factor=unknown, not enough invasion factors. Skip.\n')
  return output_string

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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
    contours,
    contour_map,
    expected_contours,
    '022 radiography'
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

  return get_string_differences(
    contours,
    contour_map,
    expected_contours,
    '022 radiography'
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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
  proximal_2 = 19
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

  return get_string_differences(
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

  return get_string_differences(
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
  metacarpal_1 = 4 
  distal_2 = 19
  medial_2 = 16
  proximal_2 = 11
  metacarpal_2 = 6
  distal_3 = 20
  medial_3 = 18
  proximal_3 = 13
  metacarpal_3 = 8
  distal_4 = 7
  medial_4 = 14
  proximal_4 = 10
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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

  return get_string_differences(
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
    first_encounter=expected_contours[0]
  )
  expected_contours[5] = ExpectedContourMedialPhalanx(
    2,
    first_encounter=expected_contours[1]
  )
  expected_contours[6] = ExpectedContourProximalPhalanx(
    2,
    first_encounter=expected_contours[2]
  )
  expected_contours[7] = ExpectedContourMetacarpal(
    2,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[4],
    first_encounter=expected_contours[3]
  )
  expected_contours[8] = ExpectedContourDistalPhalanx(
    3,
    first_encounter=expected_contours[0]
  )
  expected_contours[9] = ExpectedContourMedialPhalanx(
    3,
    first_encounter=expected_contours[1]
  )
  expected_contours[10] = ExpectedContourProximalPhalanx(
    3,
    first_encounter=expected_contours[2]
  )
  expected_contours[11] = ExpectedContourMetacarpal(
    3,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[8],
    first_encounter=expected_contours[3]
  )
  expected_contours[12] = ExpectedContourDistalPhalanx(
    4,
    first_encounter=expected_contours[0]
  )
  expected_contours[13] = ExpectedContourMedialPhalanx(
    4,
    first_encounter=expected_contours[1]
  )
  expected_contours[14] = ExpectedContourProximalPhalanx(
    4,
    first_encounter=expected_contours[2]
  )
  expected_contours[15] = ExpectedContourMetacarpal(
    4,
    ends_branchs_sequence=True,
    first_in_branch=expected_contours[12],
    first_encounter=expected_contours[3]
  )
  expected_contours[16] = ExpectedContourDistalPhalanx(
    5,
    first_encounter=expected_contours[0]
  )
  expected_contours[17] = ExpectedContourProximalPhalanx(
    5,
    first_encounter=expected_contours[2]
  )
  expected_contours[18] = ExpectedContourMetacarpal(
    5,
    first_encounter=expected_contours[3]
  )
  expected_contours[19] = ExpectedContourUlna()
  expected_contours[20] = ExpectedContourRadius()

  return expected_contours

def positional_differences_main():
  output_string = ''
  output_string = output_string + case_004_local() + '\n'
  output_string = output_string + case_022_local() + '\n'
  output_string = output_string + case_006_local() + '\n'
  output_string = output_string + case_018_local() + '\n'
  output_string = output_string + case_023_local() + '\n'
  output_string = output_string + case_029_local() + '\n'
  output_string = output_string + case_032_local() + '\n'
  output_string = output_string + case_217_local() + '\n'
  output_string = output_string + case_1622_local() + '\n'
  output_string = output_string + case_1886_local() + '\n'
  output_string = output_string + case_013_local() + '\n'
  output_string = output_string + case_016_local() + '\n'
  output_string = output_string + case_019_local() + '\n'
  output_string = output_string + case_030_local() + '\n'
  output_string = output_string + case_031_local() + '\n'
  output_string = output_string + case_084_local() + '\n'
  output_string = output_string + case_1619_local() + '\n'
  output_string = output_string + case_1779_local() + '\n'
  output_string = output_string + case_2089_local() + '\n'

  with open('positional_differences.txt', 'w') as f:
    f.write(output_string)
    print('Writing positional_differences.txt')
    print('Success.')
