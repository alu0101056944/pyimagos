'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate a table like (filename, origin, 0, 1, 2, 3, 4, 5, 6, 7, 8 ... 20) where
origin is either the segmentation selection or the actual selection made by the
search algorithm.
'''

import numpy as np

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

from src.main_experiment import main_experiment

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

def generate_sequence_differences_table_main():
  output_string = ''

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

  (
    _,
    _,
    _,
    case_to_contours_sequence_group17_5,
  ) = main_experiment(
    single=True,
    group17_5='bar',
    group18_5=None,
    group19_5=None,
    groupcontrol=None,
    nofilter=True,
    useinput2=True,
    use_starts_file=True,
    debug_mode=True,
  )

  (
    _,
    _,
    _,
    case_to_contours_sequence_group18_5,
  ) = main_experiment(
    single=True,
    group17_5=None,
    group18_5='baz',
    group19_5=None,
    groupcontrol=None,
    nofilter=True,
    useinput2=True,
    use_starts_file=True,
    debug_mode=True,
  )

  case_to_contours_sequence_group17_5.update(case_to_contours_sequence_group18_5)
  case_to_contours_sequence = case_to_contours_sequence_group17_5

  case_to_contours_sequence_indices = {}
  for case_key in case_to_contours_sequence:
    case_to_contours_sequence_indices[case_key] = []

    chosen_contours = case_to_contours_sequence[case_key]
    if len(chosen_contours) > 0:
      if case_key in case_to_manual_info:
        contours = case_to_manual_info[case_key][0]
        for chosen_contour in chosen_contours:
          for i, contour in enumerate(contours):
            if np.array_equal(chosen_contour, contour):
              case_to_contours_sequence_indices[case_key].append(i)
              break

  def table_start(encounter_amount: int = 1):
    col_specs_str = ''.join(['>{\\raggedright\\tiny\\arraybackslash}p{0.005\\textwidth}|'] * 22)
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Sequencia de contornos encontrados respecto a segmentación manual ') + (
        f'{encounter_amount}' + '}\n') + (
      '\\label{tab:sequence_differences' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|l|' + col_specs_str + '}\n') + (
      '\\toprule\n') + (
      ' & & \\multicolumn{21}{c|}{Índices de contornos} \\\\ \n') + (
      '\\hline\n \\textbf{Archivo} & \\textbf{Origen} & ') + (
          ' & '.join([str(value) for value in list(np.arange(0, 22, 1))])
        ) + ' \\\\ \n' + ('\\hline\n')
  
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

  for i, case_title in enumerate(case_to_manual_info):
    processed_title = case_title.replace('_', '\\_')
    segmentation = case_to_manual_info[case_title][1]
    
    chosen_contour_indices_str = ' & '.join(
      [str(value) for value in list(segmentation.values())]
    )
    table_entry = f'{processed_title} & manual & {chosen_contour_indices_str} \\\\ \n'

    table_cut_count += 1
    if table_cut_count >= TABLE_CUT_THRESHOLD:
      tables_generated += 1
      output_string = output_string + cut_table(table_entry,
                                                encounter_amount=tables_generated)
      table_cut_count = 0
    else:
      output_string = output_string + table_entry
    output_string = output_string + '\\hline\n'

    if case_title in case_to_contours_sequence_indices:
      computed_indices = case_to_contours_sequence_indices[case_title]
      computed_indices = [str(value) for value in computed_indices]
      padded_computed_indices = computed_indices + (
        ['-'] * max(0, 22 - len(computed_indices))
      )
      computed_chosen_contour_indices_str = ' & '.join(padded_computed_indices)
      computed_table_entry = (
        f'{processed_title} & automatic & {computed_chosen_contour_indices_str} \\\\ \n'
      )
      table_cut_count += 1
      if table_cut_count >= TABLE_CUT_THRESHOLD:
        tables_generated += 1
        output_string = output_string + cut_table(computed_table_entry,
                                                  encounter_amount=tables_generated)
        table_cut_count = 0
      else:
        output_string = output_string + computed_table_entry
      output_string = output_string + '\\hline\n'
      

  output_string = output_string + table_end()

  with open('tab_sequence_differences.txt', 'w', encoding='utf-8') as f:
    f.write(output_string)
    print('Writing tab_sequence_differences.txt')
    print('Success.')
