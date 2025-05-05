'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate a table like 
(filename,
candidate_index,
candidate_difference) to study the failure cases on radiographies that should be
sucessful in the search.
'''

import numpy as np

from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal


from src.radiographies.rad_004_with_sesamoid import (
  case_004_with_sesamoid,
  case_004_with_sesamoid_segmentation
)
from src.radiographies.rad_022_with_sesamoid import (
  case_022_with_sesamoid,
  case_022_with_sesamoid_segmentation
)
from src.radiographies.rad_006_with_sesamoid import (
  case_006_with_sesamoid,
  case_006_with_sesamoid_segmentation
)
from src.radiographies.rad_018_with_sesamoid import (
  case_018_with_sesamoid,
  case_018_with_sesamoid_segmentation
)
from src.radiographies.rad_023_with_sesamoid import (
  case_023_with_sesamoid,
  case_023_with_sesamoid_segmentation
)
from src.radiographies.rad_029_with_sesamoid import (
  case_029_with_sesamoid,
  case_029_with_sesamoid_segmentation
)
from src.radiographies.rad_032_with_sesamoid import (
  case_032_with_sesamoid,
  case_032_with_sesamoid_segmentation
)
from src.radiographies.rad_217_with_sesamoid import (
  case_217_with_sesamoid,
  case_217_with_sesamoid_segmentation
)
from src.radiographies.rad_1622_with_sesamoid import (
  case_1622_with_sesamoid,
  case_1622_with_sesamoid_segmentation
)
from src.radiographies.rad_1886_with_sesamoid import (
  case_1886_with_sesamoid,
  case_1886_with_sesamoid_segmentation
)
from src.radiographies.rad_013_with_sesamoid import (
  case_013_with_sesamoid,
  case_013_with_sesamoid_segmentation
)
from src.radiographies.rad_016_with_sesamoid import (
  case_016_with_sesamoid,
  case_016_with_sesamoid_segmentation
)
from src.radiographies.rad_019_with_sesamoid import (
  case_019_with_sesamoid,
  case_019_with_sesamoid_segmentation
)
from src.radiographies.rad_030_with_sesamoid import (
  case_030_with_sesamoid,
  case_030_with_sesamoid_segmentation
)
from src.radiographies.rad_031_with_sesamoid import (
  case_031_with_sesamoid,
  case_031_with_sesamoid_segmentation
)
from src.radiographies.rad_084_with_sesamoid import (
  case_084_with_sesamoid,
  case_084_with_sesamoid_segmentation
)
from src.radiographies.rad_1619_with_sesamoid import (
  case_1619_with_sesamoid,
  case_1619_with_sesamoid_segmentation
)
from src.radiographies.rad_1779_with_sesamoid import (
  case_1779_with_sesamoid,
  case_1779_with_sesamoid_segmentation
)
from src.radiographies.rad_2089_with_sesamoid import (
  case_2089_with_sesamoid,
  case_2089_with_sesamoid_segmentation
)

from src.main_experiment_penalization import all_case_tuples

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

def generate_case_identification_main():
  case_to_manual_info = {
    '004_borders_clean.jpg': [case_004_with_sesamoid(), case_004_with_sesamoid_segmentation()],
    '022_borders_clean.jpg': [case_022_with_sesamoid(), case_022_with_sesamoid_segmentation()],
    '006_borders_clean.jpg': [case_006_with_sesamoid(), case_006_with_sesamoid_segmentation()],
    '018_borders_clean.jpg': [case_018_with_sesamoid(), case_018_with_sesamoid_segmentation()],
    '023_borders_clean.jpg': [case_023_with_sesamoid(), case_023_with_sesamoid_segmentation()],
    '029_borders_clean.jpg': [case_029_with_sesamoid(), case_029_with_sesamoid_segmentation()],
    '032_borders_clean.jpg': [case_032_with_sesamoid(), case_032_with_sesamoid_segmentation()],
    '217_borders_clean.jpg': [case_217_with_sesamoid(), case_217_with_sesamoid_segmentation()],
    '1622_borders_clean.jpg': [case_1622_with_sesamoid(), case_1622_with_sesamoid_segmentation()],
    '1886_borders_clean.jpg': [case_1886_with_sesamoid(), case_1886_with_sesamoid_segmentation()],
    '013_borders_clean.jpg': [case_013_with_sesamoid(), case_013_with_sesamoid_segmentation()],
    '016_borders_clean.jpg': [case_016_with_sesamoid(), case_016_with_sesamoid_segmentation()],
    '019_borders_clean.jpg': [case_019_with_sesamoid(), case_019_with_sesamoid_segmentation()],
    '030_borders_clean.jpg': [case_030_with_sesamoid(), case_030_with_sesamoid_segmentation()],
    '031_borders_clean.jpg': [case_031_with_sesamoid(), case_031_with_sesamoid_segmentation()],
    '084_borders_clean.jpg': [case_084_with_sesamoid(), case_084_with_sesamoid_segmentation()],
    '1619_borders_clean.jpg': [case_1619_with_sesamoid(), case_1619_with_sesamoid_segmentation()],
    '1779_borders_clean.jpg': [case_1779_with_sesamoid(), case_1779_with_sesamoid_segmentation()],
    '2089_borders_clean.jpg': [case_2089_with_sesamoid(), case_2089_with_sesamoid_segmentation()],
  }

  # minimize the manual contours to match the automatic ones
  for case in case_to_manual_info:
    case_to_manual_info[case][0] = minimize_contours(case_to_manual_info[case][0])

    
  informal_case_name_to_filename = {
    'case_004_distal2': '004_borders_clean.jpg',
    'case_004_distal2': '004_borders_clean.jpg',
    'case_004_metacarpal1': '004_borders_clean.jpg',
    'case_006_distal2': '006_borders_clean.jpg',
    'case_023_distal2': '023_borders_clean.jpg',
    'case_023_distal5': '023_borders_clean.jpg',
    'case_022_distal5': '022_borders_clean.jpg',
  }

  output_string = ''
  def table_start(encounter_amount: int = 1):
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Contornos involucrados en la selección de contorno esperado ') + (
        f'{encounter_amount}' + '}\n') + (
      '\\label{tab:case_identification ' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|c|c|}\n') + (
      '\\toprule\n') + (
      '\\textbf{Caso} & \\textbf{Índice del contorno} & \\textbf{Diferencia} \\\\ \n') + (
      '\\hline\n')
  
  def table_end():
    return (
      '\\bottomrule\n') + (
      '\\end{tabular}\n') + (
      '\\end{footnotesize}\n') + (
      '\\end{table}\n')
  
  def cut_table(new_line: str, encounter_amount):
    return table_end() + '\n' + table_start(encounter_amount) + new_line

  output_string = output_string + table_start()

  expected_contour_to_cases = all_case_tuples()

  tables_generated = 1
  table_cut_count = 0
  TABLE_CUT_THRESHOLD = 12

  for expected_contour_key in expected_contour_to_cases:
    for case_info in expected_contour_to_cases[expected_contour_key]:
      target_expected_contour = case_info[0]
      candidate_contours = case_info[1]
      case_title = case_info[3]

      processed_title = case_title.replace('_', '\\_')
    
      filename = informal_case_name_to_filename[case_title]
      contours = case_to_manual_info[filename][0]

      candidate_contour_indices = []
      for candidate_contour in candidate_contours:
        for i, contour in enumerate(contours):
          if np.array_equal(candidate_contour, contour):
            candidate_contour_indices.append(i)
            break

      scores = []
      for i, candidate_contour in enumerate(candidate_contours):
        target_expected_contour.prepare(
          candidate_contour,
          image_width=301,
          image_height=462,
        )
        score = target_expected_contour.shape_restrictions()
        scores.append(score)
      
      for i in range(len(candidate_contours)):
        table_entry = f'{processed_title} & {candidate_contour_indices[i]}' + (
          f' & {scores[i]} \\\\ \n')

        table_cut_count += 1
        if table_cut_count >= TABLE_CUT_THRESHOLD:
          tables_generated += 1
          output_string = output_string + cut_table(table_entry,
                                                    encounter_amount=tables_generated)
          table_cut_count = 0
        else:
          output_string = output_string + table_entry
        if i == len(candidate_contours) - 1:
          output_string = output_string + '\\midrule\n'
          output_string = output_string + '\\hline\n'
        else:
          output_string = output_string + '\\hline\n'

  output_string = output_string + table_end()

  with open('tab_case_identification.txt', 'w', encoding='utf-8') as f:
    f.write(output_string)
    print('Writing tab_case_identification.txt')
    print('Success.')
