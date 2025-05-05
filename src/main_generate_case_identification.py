'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate a table like (

filename,
expected_contour,
candidate_index,
candidate_difference,

involved_indices) to study the failure cases on radiographies that should be
sucessful in the search.
'''

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

def generate_case_identification_main():
  output_string = ''
  def table_start(encounter_amount: int = 1):
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Contornos involucrados en la selección de contorno esperado ') + (
        f'{encounter_amount}' + '}\n') + (
      '\\label{tab:case_identification ' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|l|c|c|}\n') + (
      '\\toprule\n') + (
      '\\textbf{Archivo} & \\textbf{Contorno esperado} & \\textbf{Índice del contorno} & \\textbf{Diferencia} \\\\ \n') + (
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
