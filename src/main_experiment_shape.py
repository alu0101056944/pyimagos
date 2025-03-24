'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Print per contour-shape the shape score and the reason it returned infinite if
applies.
'''

import copy
import json

import numpy as np

from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.proximal_phalanx import ExpectedContourProximalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from src.expected_contours.ulna import ExpectedContourUlna
from src.expected_contours.radius import ExpectedContourRadius
from src.expected_contours.metacarpal_sesamoid import (
  ExpectedContourSesamoidMetacarpal
)
from src.expected_contours.sesamoid import ExpectedContourSesamoid

from src.radiographies.rad_004 import case_004, case_004_segmentation
from src.radiographies.rad_004_with_sesamoid import (
  case_004_with_sesamoid,
  case_004_with_sesamoid_segmentation
)
from src.radiographies.rad_022 import case_022, case_022_segmentation
from src.radiographies.rad_022_with_sesamoid import (
  case_022_with_sesamoid,
  case_022_with_sesamoid_segmentation
)
from src.radiographies.rad_006 import case_006, case_006_segmentation
from src.radiographies.rad_006_with_sesamoid import (
  case_006_with_sesamoid,
  case_006_with_sesamoid_segmentation
)
from src.radiographies.rad_018 import case_018, case_018_segmentation
from src.radiographies.rad_018_with_sesamoid import (
  case_018_with_sesamoid,
  case_018_with_sesamoid_segmentation
)
from src.radiographies.rad_023 import case_023, case_023_segmentation
from src.radiographies.rad_023_with_sesamoid import (
  case_023_with_sesamoid,
  case_023_with_sesamoid_segmentation
)
from src.radiographies.rad_029 import case_029, case_029_segmentation
from src.radiographies.rad_029_with_sesamoid import (
  case_029_with_sesamoid,
  case_029_with_sesamoid_segmentation
)
from src.radiographies.rad_032 import case_032, case_032_segmentation
from src.radiographies.rad_032_with_sesamoid import (
  case_032_with_sesamoid,
  case_032_with_sesamoid_segmentation
)
from src.radiographies.rad_217 import case_217, case_217_segmentation
from src.radiographies.rad_217_with_sesamoid import (
  case_217_with_sesamoid,
  case_217_with_sesamoid_segmentation
)
from src.radiographies.rad_1622 import case_1622, case_1622_segmentation
from src.radiographies.rad_1622_with_sesamoid import (
  case_1622_with_sesamoid,
  case_1622_with_sesamoid_segmentation
)
from src.radiographies.rad_1886 import case_1886, case_1886_segmentation
from src.radiographies.rad_1886_with_sesamoid import (
  case_1886_with_sesamoid,
  case_1886_with_sesamoid_segmentation
)
from src.radiographies.rad_013 import case_013, case_013_segmentation
from src.radiographies.rad_013_with_sesamoid import (
  case_013_with_sesamoid,
  case_013_with_sesamoid_segmentation
)
from src.radiographies.rad_016 import case_016, case_016_segmentation
from src.radiographies.rad_016_with_sesamoid import (
  case_016_with_sesamoid,
  case_016_with_sesamoid_segmentation
)
from src.radiographies.rad_019 import case_019, case_019_segmentation
from src.radiographies.rad_019_with_sesamoid import (
  case_019_with_sesamoid,
  case_019_with_sesamoid_segmentation
)
from src.radiographies.rad_030 import case_030, case_030_segmentation
from src.radiographies.rad_030_with_sesamoid import (
  case_030_with_sesamoid,
  case_030_with_sesamoid_segmentation
)
from src.radiographies.rad_031 import case_031, case_031_segmentation
from src.radiographies.rad_031_with_sesamoid import (
  case_031_with_sesamoid,
  case_031_with_sesamoid_segmentation
)
from src.radiographies.rad_084 import case_084, case_084_segmentation
from src.radiographies.rad_084_with_sesamoid import (
  case_084_with_sesamoid,
  case_084_with_sesamoid_segmentation
)
from src.radiographies.rad_1619 import case_1619, case_1619_segmentation
from src.radiographies.rad_1619_with_sesamoid import (
  case_1619_with_sesamoid,
  case_1619_with_sesamoid_segmentation
)
from src.radiographies.rad_1779 import case_1779, case_1779_segmentation
from src.radiographies.rad_1779_with_sesamoid import (
  case_1779_with_sesamoid,
  case_1779_with_sesamoid_segmentation
)
from src.radiographies.rad_2089 import case_2089, case_2089_segmentation
from src.radiographies.rad_2089_with_sesamoid import (
  case_2089_with_sesamoid,
  case_2089_with_sesamoid_segmentation
)
from constants import CRITERIA_DICT

def write_expected_contours_precisions_stage_1():
  output_string = ''
  deltas = {
    'distal': {
      'area': [2, 1, 0, -1, -2],
      'aspect_ratio_min': list(np.arange(1.6, -0.8, -0.1)),
      'aspect_ratio_max': list(np.arange(1.6, -0.8, -0.1)),
      'aspect_ratio_tolerance': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': [0.04, 0.03, 0.02, 0.01, 0, -0.01, -0.02, -0.03, -0.04]
    },
    'medial': {
      'area': [2, 1, 0, -1, -2],
      'aspect_ratio_min': list(np.arange(0.8, -1.6, -0.1)),
      'aspect_ratio_max': list(np.arange(0.8, -1.6, -0.1)),
      'aspect_ratio_tolerance': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': list(np.arange(0.8, -0.8, -0.01))
    },
    'proximal': {
      'area': [2, 1, 0, -1, -2],
      'aspect_ratio_min': list(np.arange(1.6, -2, -0.1)),
      'aspect_ratio_max': list(np.arange(1.6, -2, -0.1)),
      'aspect_ratio_tolerance': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': [0.04, 0.03, 0.02, 0.01, 0, -0.01, -0.02, -0.03, -0.04]
    },
    'metacarpal': {
      'area': [2, 1, 0, -1, -2],
      'aspect_ratio_min': list(np.arange(2.6, -2.6, -0.1)),
      'aspect_ratio_max': list(np.arange(2.6, -2.6, -0.1)),
      'aspect_ratio_tolerance': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': [0.04, 0.03, 0.02, 0.01, 0, -0.01, -0.02, -0.03, -0.04]
    },
    'radius': {
      'area': [30, 20, 10, 0, -10, -20, -30],
      'aspect_ratio_min': list(np.arange(0.3, -1.6, -0.1)),
      'aspect_ratio_max': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': list(np.arange(1.6, -1.6, -0.01))
    },
    'ulna': {
      'area': [30, 20, 10, 0, -10, -20, -30],
      'aspect_ratio_min': list(np.arange(1.3, -1.3, -0.1)),
      'aspect_ratio_max': list(np.arange(1.3, -1.3, -0.1)),
      'solidity': [0.15, 0.10, 0.05, 0, -0.05, -0.1, -0.15],
      'defect_area_ratio': [
        0.004,
        0.003,
        0.002,
        0.001,
        0,
        -0.001,
        -0.002,
        -0.003,
        -0.004
      ]
    },
  }

  incorrect_expected_contours = {
    'distal': {
      'area': [],
      'aspect_ratio_min': [],
      'aspect_ratio_max': ['medial'],
      'aspect_ratio_tolerance': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
    'medial': {
      'area': [],
      'aspect_ratio_min': ['distal'],
      'aspect_ratio_max': ['proximal'],
      'aspect_ratio_tolerance': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
    'proximal': {
      'area': [],
      'aspect_ratio_min': ['medial'],
      'aspect_ratio_max': ['metacarpal'],
      'aspect_ratio_tolerance': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
    'metacarpal': {
      'area': [],
      'aspect_ratio_min': ['proximal'],
      'aspect_ratio_max': [],
      'aspect_ratio_tolerance': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
    'radius': {
      'area': [],
      'aspect_ratio_min': [],
      'aspect_ratio_max': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
    'ulna': {
      'area': [],
      'aspect_ratio_min': [],
      'aspect_ratio_max': [],
      'solidity': [],
      'defect_area_ratio': [],
    },
  }

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

  (
    precisions,
    best_precisions,
    best_factors,
  ) = get_results(deltas, all_contours, expected_contours,
                  incorrect_expected_contours)

  output_string = output_string + '#Deltas:\n'
  deltas_string = json.dumps(deltas, indent=2)
  output_string = output_string + deltas_string + '\n\n'

  output_string = output_string + '#Precisions obtained:\n'
  precisions_string = json.dumps(precisions, indent=2)
  output_string = output_string + precisions_string + '\n\n'
  
  output_string = output_string + '#Best precisions:\n'
  best_precisions_string = json.dumps(best_precisions, indent=2)
  output_string = output_string + best_precisions_string + '\n\n'

  output_string = output_string + '#Best factors:\n'
  best_factors_string = json.dumps(best_factors, indent=2)
  output_string = output_string + best_factors_string + '\n'

  with open('best_shape_factors.txt', 'w') as f:
    f.write(output_string)
    print('Writing best_shape_factors.txt')
    print('Success.')

def write_expected_contours_precisions_stage_2():
  output_string = ''
  deltas = {
    'sesamoid': {
      'solidity': [0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3],
    },
  }
  incorrect_expected_contours = {
    'sesamoid': {
      'solidity': [],
    },
  }

  all_contours = [
    [case_004_with_sesamoid(), case_004_with_sesamoid_segmentation()],
    [case_022_with_sesamoid(), case_022_with_sesamoid_segmentation()],
    [case_006_with_sesamoid(), case_006_with_sesamoid_segmentation()],
    [case_018_with_sesamoid(), case_018_with_sesamoid_segmentation()],
    [case_023_with_sesamoid(), case_023_with_sesamoid_segmentation()],
    [case_029_with_sesamoid(), case_029_with_sesamoid_segmentation()],
    [case_032_with_sesamoid(), case_032_with_sesamoid_segmentation()],
    [case_217_with_sesamoid(), case_217_with_sesamoid_segmentation()],
    [case_1622_with_sesamoid(), case_1622_with_sesamoid_segmentation()],
    [case_1886_with_sesamoid(), case_1886_with_sesamoid_segmentation()],
    [case_013_with_sesamoid(), case_013_with_sesamoid_segmentation()],
    [case_016_with_sesamoid(), case_016_with_sesamoid_segmentation()],
    [case_019_with_sesamoid(), case_019_with_sesamoid_segmentation()],
    [case_030_with_sesamoid(), case_030_with_sesamoid_segmentation()],
    [case_031_with_sesamoid(), case_031_with_sesamoid_segmentation()],
    [case_084_with_sesamoid(), case_084_with_sesamoid_segmentation()],
    [case_1619_with_sesamoid(), case_1619_with_sesamoid_segmentation()],
    [case_1779_with_sesamoid(), case_1779_with_sesamoid_segmentation()],
    [case_2089_with_sesamoid(), case_2089_with_sesamoid_segmentation()],
  ]

  # change contours to only metacarpal and sesamoid, also segmentation.
  for contours_info in all_contours:
    contours = contours_info[0]
    segmentation = contours_info[1]
    metacarpal_5_index = segmentation['metacarpal_5']
    sesamoid_index = segmentation['sesamoid']
    contours_info[0] = [
      contours[i] for i in [metacarpal_5_index, sesamoid_index]
    ]
    contours_info[1] = {
      'metacarpal_5': 0,
      'sesamoid': 1,
    }

  expected_contours = get_canonical_expected_contours_stage_2()

  (
    precisions,
    best_precisions,
    best_factors,
  ) = get_results(deltas, all_contours, expected_contours,
                  incorrect_expected_contours)

  output_string = output_string + '#Deltas:\n'
  deltas_string = json.dumps(deltas, indent=2)
  output_string = output_string + deltas_string + '\n\n'

  output_string = output_string + '#Precisions obtained:\n'
  precisions_string = json.dumps(precisions, indent=2)
  output_string = output_string + precisions_string + '\n\n'
  
  output_string = output_string + '#Best precisions:\n'
  best_precisions_string = json.dumps(best_precisions, indent=2)
  output_string = output_string + best_precisions_string + '\n\n'

  output_string = output_string + '#Best factors:\n'
  best_factors_string = json.dumps(best_factors, indent=2)
  output_string = output_string + best_factors_string + '\n'

  with open('best_shape_factors_stage_2.txt', 'w') as f:
    f.write(output_string)
    print('Writing best_shape_factors_stage_2.txt')
    print('Success.')

def get_results(deltas: dict, all_contours: list, expected_contours: list,
                incorrect_expected_contours: dict):
  # dict of expected_contour-factor-to array of precisions (one per delta)
  precisions = copy.deepcopy(deltas)

  for expected_contour_key in deltas:
    for factor_key in deltas[expected_contour_key]:
      precisions[expected_contour_key][factor_key] = []

      for delta in deltas[expected_contour_key][factor_key]:
        criteria_dict = copy.deepcopy(CRITERIA_DICT)
        criteria_dict[expected_contour_key][factor_key] = (
          criteria_dict[expected_contour_key][factor_key] + delta
        )

        success_list = []
        for i, contours_info in enumerate(all_contours):
          contours = contours_info[0]
          segmentation = contours_info[1]

          relevant_contour_info = [
            (i, value)
            for i, (key, value) in enumerate(segmentation.items())
            if key.startswith(expected_contour_key)
          ]

          for contour_info in relevant_contour_info:
            contour = contours[contour_info[1]]
            expected_contour = expected_contours[contour_info[0]]
            
            points = np.reshape(contour, (-1, 2))
            if len(points) == 0:
              raise ValueError(f'Found empty contour at rad. index={i}')
            
            all_x = points[:, 0]
            all_y = points[:, 1]
            min_x = np.min(all_x)
            max_x = np.max(all_x)
            min_y = np.min(all_y)
            max_y = np.max(all_y)
            image_width = int(max_x - min_x)
            image_height = int(max_y - min_y)
            expected_contour.prepare(contour, image_width, image_height)

            hu_moment, factorname_to_info = (
              expected_contour.shape_restrictions(criteria=criteria_dict,
                                                  decompose=True)
            )

            if factor_key in factorname_to_info:
              if factorname_to_info[factor_key]['fail_status'] == True:
                success_list.append(False)
              else:
                success_list.append(True)

          # To test other expected contours with the new (factor + delta) value
          # so that a penalization happens if the new factor is generalizing too
          # much, those other "incorrect" expected contours are taken into
          # account in the fail rate. If an expected contour of the incorrect, which
          # are used as incorrect reference fail, the final precision must go
          # down as penalization, so reverse the fail result of the incorrect
          # expected contour.

          # Example: ['distal', 'medial']
          incorrect_expected_contours_local = (
            incorrect_expected_contours[expected_contour_key][factor_key]
          )

          for expected_contour_key_2 in incorrect_expected_contours_local:
            if expected_contour_key_2 == expected_contour_key:
              raise ValueError('Specified same key as incorrect key.')

            original_factor = criteria_dict[expected_contour_key_2][factor_key]
            new_factor = criteria_dict[expected_contour_key][factor_key]
            criteria_dict[expected_contour_key_2][factor_key] = new_factor

            relevant_contour_indices_incorrect_2 = [
              (i, value)
              for i, (key, value) in enumerate(segmentation.items())
              if key.startswith(expected_contour_key_2)
            ]

            for contour_info in relevant_contour_indices_incorrect_2:
              contour = contours[contour_info[1]]
              expected_contour = expected_contours[contour_info[0]]

              points = np.reshape(contour, (-1, 2))
              if len(points) == 0:
                raise ValueError(f'Found empty contour at rad. index={i}')
              
              all_x = points[:, 0]
              all_y = points[:, 1]
              min_x = np.min(all_x)
              max_x = np.max(all_x)
              min_y = np.min(all_y)
              max_y = np.max(all_y)
              image_width = int(max_x - min_x)
              image_height = int(max_y - min_y)
              expected_contour.prepare(contour, image_width, image_height)

              hu_moment, factorname_to_info = (
                expected_contour.shape_restrictions(criteria=criteria_dict,
                                                    decompose=True)
              )

              if factor_key in factorname_to_info:
                if factorname_to_info[factor_key]['fail_status'] == True:
                  success_list.append(True)
                else:
                  success_list.append(False)

            # Put the original factor back in, testing is done.
            criteria_dict[expected_contour_key_2][factor_key] = original_factor

        positive_amount = success_list.count(True)

        if len(success_list) > 0:
          precision = positive_amount / len(success_list)
          precisions[expected_contour_key][factor_key].append(precision)
        else:
          precisions[expected_contour_key][factor_key].append(float('-inf'))
  
  best_precisions = {}
  best_factors = {}
  for expected_contour_key in precisions:
    best_precisions[expected_contour_key] = {}
    best_factors[expected_contour_key] = {}
    for factor_key in precisions[expected_contour_key]:
      local_precisions = precisions[expected_contour_key][factor_key]

      # flip so that it returns the last max equal occurence and not the latest.
      best_precision_index = np.argmax(np.flip(local_precisions))
      best_precision = (
        precisions[expected_contour_key][factor_key][best_precision_index]
      )

      best_precisions[expected_contour_key][factor_key] = best_precision

      best_delta = (
        deltas[expected_contour_key][factor_key][best_precision_index]
      )
      original_value = criteria_dict[expected_contour_key][factor_key]

      best_factors[expected_contour_key][factor_key] = (
        original_value + best_delta
      )

  return precisions, best_precisions, best_factors

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

def get_canonical_expected_contours_stage_2():
  expected_contours = [
    ExpectedContourSesamoidMetacarpal(),
    ExpectedContourSesamoid()
  ]
  return expected_contours

def write_shape_experiment():
  write_expected_contours_precisions_stage_1()
  write_expected_contours_precisions_stage_2()
