'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Given a list of (expected_contour, position_restriction, invasion_count) calculate
the precision for different adaptations of the position restrictions, display the
precision-pair results highlighting the best assignments.

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
from src.radiographies.rad_004 import case_004, case_004_segmentation
from src.radiographies.rad_022 import case_022, case_022_segmentation
from src.radiographies.rad_006 import case_006, case_006_segmentation
from src.radiographies.rad_018 import case_018, case_018_segmentation
from src.radiographies.rad_023 import case_023, case_023_segmentation
from src.radiographies.rad_029 import case_029, case_029_segmentation
from src.radiographies.rad_032 import case_032, case_032_segmentation
from src.radiographies.rad_217 import case_217, case_217_segmentation
from src.radiographies.rad_1622 import case_1622, case_1622_segmentation
from src.radiographies.rad_1886 import case_1886, case_1886_segmentation
from src.radiographies.rad_013 import case_013, case_013_segmentation
from src.radiographies.rad_016 import case_016, case_016_segmentation
from src.radiographies.rad_019 import case_019, case_019_segmentation
from src.radiographies.rad_030 import case_030, case_030_segmentation
from src.radiographies.rad_031 import case_031, case_031_segmentation
from src.radiographies.rad_084 import case_084, case_084_segmentation
from src.radiographies.rad_1619 import case_1619, case_1619_segmentation
from src.radiographies.rad_1779 import case_1779, case_1779_segmentation
from src.radiographies.rad_2089 import case_2089, case_2089_segmentation

def count_invasion_factor_max(contour: np.array,
                          position_restrictions: list) -> list[float]:
  '''The invasion factor is the distance from the reference line when on the
  wrong side. Get the point that is furthest away from the reference line.'''

  invasion_factors = []
  for position_restriction in position_restrictions:
    p1, p2, allowed_side_array = position_restriction
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    furthest_invasion_factor = float('-inf')
    if x2 == x1:
      allowed_side = allowed_side_array[3]
      x_line = x1

      for point in contour:
        x = point[0][0]
        if allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER:
          if x <= x_line:
            local_invasion_factor = (x_line - x)
            if local_invasion_factor > furthest_invasion_factor:
              furthest_invasion_factor = local_invasion_factor
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
          if x >= x_line:
            local_invasion_factor = (x - x_line)
            if local_invasion_factor > furthest_invasion_factor:
              furthest_invasion_factor = local_invasion_factor
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
          if x < x_line:
            local_invasion_factor = (x_line - x)
            if local_invasion_factor > furthest_invasion_factor:
              furthest_invasion_factor = local_invasion_factor
        elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
          if x > x_line:
            local_invasion_factor = (x - x_line)
            if local_invasion_factor > furthest_invasion_factor:
              furthest_invasion_factor = local_invasion_factor
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
                local_invasion_factor = (line_y - y_point)
                if local_invasion_factor > furthest_invasion_factor:
                  furthest_invasion_factor = local_invasion_factor
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER:
              if y_point >= line_y:
                local_invasion_factor = (y_point - line_y)
                if local_invasion_factor > furthest_invasion_factor:
                  furthest_invasion_factor = local_invasion_factor
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL:
              if y_point < line_y:
                local_invasion_factor = (line_y - y_point)
                if local_invasion_factor > furthest_invasion_factor:
                  furthest_invasion_factor = local_invasion_factor
            elif allowed_side == AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL:
              if y_point > line_y:
                local_invasion_factor = (y_point - line_y)
                if local_invasion_factor > furthest_invasion_factor:
                  furthest_invasion_factor = local_invasion_factor
    
    invasion_factors.append(furthest_invasion_factor)

  return invasion_factors


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

def get_all_invasion_factors(contours: list, contour_map: list,
                             expected_contours: list) -> list:
  '''contour_map is the ordered version of contours which corresponds to the
  expected_contours sequence, which is a list of ExpectedContour classes.'''
  all_invasion_factors = []
  
  for j in range(len(contour_map)):
    contour = contours[contour_map[j]]
    expected_contour = expected_contours[j]
    
    points = np.reshape(contour, (-1, 2))
    if len(points) == 0:
      raise ValueError('Empty contour as expected contour.')
    
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
    all_invasion_factors.append(invasion_factors)

  return all_invasion_factors

def write_position_experiment_first_stage():
  all_contours = [
    [case_004(), case_004_segmentation()],
    [case_022(), case_022_segmentation()],
    [case_006(), case_006_segmentation()],
    [case_018(), case_018_segmentation()],
    [case_023(), case_023_segmentation()],
    [case_029(), case_029_segmentation()],
    [case_032(), case_032_segmentation()],
    [case_217(), case_217_segmentation()],
    [case_1622(), case_1622_segmentation()],
    [case_1886(), case_1886_segmentation()],
    [case_013(), case_013_segmentation()],
    [case_016(), case_016_segmentation()],
    [case_019(), case_019_segmentation()],
    [case_030(), case_030_segmentation()],
    [case_031(), case_031_segmentation()],
    [case_084(), case_084_segmentation()],
    [case_1619(), case_1619_segmentation()],
    [case_1779(), case_1779_segmentation()],
    [case_2089(), case_2089_segmentation()],
  ]

  expected_contours = get_canonical_expected_contours()

  if len(all_contours) == 0:
    raise ValueError('No contours cases, need at least one.')

  # calculate furthest on each position restriction
  all_global_furthest_invasion = None
  for contours_info in all_contours:
    contours = contours_info[0]
    contour_map = list(contours_info[1].values())

    all_local_furthest_invasion = get_all_invasion_factors(
      contours,
      contour_map,
      expected_contours,
    )

    if all_global_furthest_invasion is None:
      all_global_furthest_invasion = all_local_furthest_invasion
    else:
      for i, expected_invasions in enumerate(all_global_furthest_invasion):
        for j, restriction_invasion in enumerate(expected_invasions):
          new_invasion = all_local_furthest_invasion[i][j]
          old_invasion = restriction_invasion
          if new_invasion > old_invasion:
            all_global_furthest_invasion[i][j] = new_invasion

  output_string = 'Restrictions: global furthest invasions:\n'
  for i, expected_invasions in enumerate(all_global_furthest_invasion):
    output_string = output_string + f'R{i}: '
    for j, restriction_invasion in enumerate(expected_invasions):
      output_string = output_string + f'{restriction_invasion}' + (
          f'{"," if j != len(expected_invasions) - 1 else ""}')
    output_string = output_string + f'\n'

  with open('global_furthest_invasions.txt', 'w') as f:
    f.write(output_string)
    print('Writing global_furthest_invasions.txt')
    print('Success.')

def write_position_experiment():
  write_position_experiment_first_stage()

