'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate the table that represents all 19 (radiographies) * 22 contours.
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

def enconde_smallest_str(contour):
  contour_reshaped = np.reshape(contour, (-1, 2))
  return ";".join(f"({x},{y})" for x, y in contour_reshaped)

def generate_contours_table_main():
  output_string = ''

  all_contours = [
    ['004_borders_clean.jpg', [case_004_with_sesamoid(), case_004_with_sesamoid_segmentation()]],
    ['022_borders_clean.jpg', [case_022_with_sesamoid(), case_022_with_sesamoid_segmentation()]],
    ['006_borders_clean.jpg', [case_006_with_sesamoid(), case_006_with_sesamoid_segmentation()]],
    ['018_borders_clean.jpg', [case_018_with_sesamoid(), case_018_with_sesamoid_segmentation()]],
    ['023_borders_clean.jpg', [case_023_with_sesamoid(), case_023_with_sesamoid_segmentation()]],
    ['029_borders_clean.jpg', [case_029_with_sesamoid(), case_029_with_sesamoid_segmentation()]],
    ['032_borders_clean.jpg', [case_032_with_sesamoid(), case_032_with_sesamoid_segmentation()]],
    ['217_borders_clean.jpg', [case_217_with_sesamoid(), case_217_with_sesamoid_segmentation()]],
    ['1622_borders_clean.jpg', [case_1622_with_sesamoid(), case_1622_with_sesamoid_segmentation()]],
    ['1886_borders_clean.jpg', [case_1886_with_sesamoid(), case_1886_with_sesamoid_segmentation()]],
    ['013_borders_clean.jpg', [case_013_with_sesamoid(), case_013_with_sesamoid_segmentation()]],
    ['016_borders_clean.jpg', [case_016_with_sesamoid(), case_016_with_sesamoid_segmentation()]],
    ['019_borders_clean.jpg', [case_019_with_sesamoid(), case_019_with_sesamoid_segmentation()]],
    ['030_borders_clean.jpg', [case_030_with_sesamoid(), case_030_with_sesamoid_segmentation()]],
    ['031_borders_clean.jpg', [case_031_with_sesamoid(), case_031_with_sesamoid_segmentation()]],
    ['084_borders_clean.jpg', [case_084_with_sesamoid(), case_084_with_sesamoid_segmentation()]],
    ['1619_borders_clean.jpg', [case_1619_with_sesamoid(), case_1619_with_sesamoid_segmentation()]],
    ['1779_borders_clean.jpg', [case_1779_with_sesamoid(), case_1779_with_sesamoid_segmentation()]],
    ['2089_borders_clean.jpg', [case_2089_with_sesamoid(), case_2089_with_sesamoid_segmentation()]],
  ]

  def table_start(encounter_amount: int = 1):
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Contornos de cada archivo ' + f'{encounter_amount}' + '}\n') + (
      '\\label{tab:all_contours' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|c|>{\\tiny}p{0.6\\textwidth}|}\n') + (
      '\\toprule\n') + (
      '\\textbf{Archivo} & \\textbf{' + 'Índice} & \\multicolumn{1}{l|}{\\textbf{Coordenadas}} \\\\ \n')
  
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
  TABLE_CUT_THRESHOLD = 3

  CONTOUR_LINE_JUMP_THRESHOLD = 10

  for contour_info in all_contours:
    title = contour_info[0].replace('_', '\\_')
    contours = contour_info[1][0]
    segmentation = contour_info[1][1]
    
    for i, segment_key in enumerate(segmentation):
      contour_index = segmentation[segment_key]
      contour_small_str = enconde_smallest_str(contours[contour_index])

      contour_small_str_chunks = contour_small_str.split(';')
      contour_str_array = []
      for j, chunk in enumerate(contour_small_str_chunks):
        contour_str_array.append(chunk)
        if (j + 1) % CONTOUR_LINE_JUMP_THRESHOLD == 0 and (j + 1) < len(contour_small_str_chunks):
          contour_str_array.append('\\par')
        elif (j + 1) < len(contour_small_str_chunks):
          contour_str_array.append(';')
      contour_str = ''.join(contour_str_array)

      table_cut_count += 1
      if table_cut_count >= TABLE_CUT_THRESHOLD:
        tables_generated += 1
        output_string = output_string + cut_table(f'{title} & {i} & {contour_str} \\\\ \n',
                                                  encounter_amount=tables_generated)
        if i == len(segmentation) - 1:
          output_string = output_string + '\\midrule\n'
        else:
          output_string = output_string + '\\hline\n'
        table_cut_count = 0
        
      else:
        output_string = output_string + f'{title} & {i} & {contour_str} \\\\ \n'
        if i == len(segmentation) - 1:
          output_string = output_string + '\\midrule\n'
        else:
          output_string = output_string + '\\hline\n'

  output_string = output_string + table_end()

  with open('tab_all_contours.txt', 'w', encoding='utf-8') as f:
    f.write(output_string)
    print('Writing tab_all_contours.txt')
    print('Success.')
