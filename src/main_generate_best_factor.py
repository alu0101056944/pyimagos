'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Generate a table like (contour_type, best_factor, precision) to study
the failure cases on radiographies that should be sucessful in the search.
'''

import copy

import numpy as np

from constants import CRITERIA_DICT

from src.main_experiment_penalization import all_case_tuples

def generate_best_factor_main():
  expected_contour_to_cases = all_case_tuples()

  output_string = ''
  def table_start(encounter_amount: int = 1):
    return (
      '\\begin{table}[H]\n') + (
      '\\centering\n') + (
      '\\caption{Mejor factor de penalización por tipo de contorno ') + (
        f'{encounter_amount}' + '}\n') + (
      '\\label{tab:best_factor ' + f'{encounter_amount}' + '}\n') + (
      '\\begin{footnotesize}\n') + (
      '\\begin{tabular}{|l|c|c|}\n') + (
      '\\toprule\n') + (
      '\\textbf{Tipo de contorno} & \\textbf{Factor de penalización} & \\textbf{Precisión} \\\\ \n') + (
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

  criteria_dict = copy.deepcopy(CRITERIA_DICT)
  step = 0.0025
  range = 40

  expected_contour_to_cases = all_case_tuples()

  expected_contour_to_case_to_contour_to_difference = {}
  for expected_contour_key in expected_contour_to_cases:
    expected_contour_to_case_to_contour_to_difference[expected_contour_key] = {}
    for case_info in expected_contour_to_cases[expected_contour_key]:
      target_expected_contour = case_info[0]
      candidate_contours = case_info[1]
      correct_candidate_index = case_info[2]
      case_title = case_info[3]
      
      expected_contour_to_case_to_contour_to_difference[
        expected_contour_key][case_title] = {}
      expected_contour_to_case_to_contour_to_difference[
        expected_contour_key][case_title]['original'] = {
        'correct_candidate_index': correct_candidate_index,
        'chosen_candidate_index': None,
        'candidate_contour_differences': []
      }

      scores = []
      for i, candidate_contour in enumerate(candidate_contours):
        target_expected_contour.prepare(
          candidate_contour,
          image_width=301,
          image_height=462,
        )
        score = target_expected_contour.shape_restrictions(criteria_dict)
        scores.append(score)

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title]['original']['candidate_contour_differences'].append(score)

        chosen_candidate_index = int(np.argmin(scores))
        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title]['original']['chosen_candidate_index'] = chosen_candidate_index

  expected_contour_to_factor_to_precision = {}
  for expected_contour_key in expected_contour_to_cases:
    expected_contour_to_factor_to_precision[expected_contour_key] = {}

  for expected_contour_key in expected_contour_to_cases:
    contour_type = expected_contour_key.split('_')[0]
    
    first_stage_penalization_factors = list(np.arange(1.0, 0.1, -0.1))
    first_stage_penalization_factors = [
      float(factor) for factor in first_stage_penalization_factors
    ]

    penalization_to_success_list = {}
    for penalization_factor in first_stage_penalization_factors:
      penalization_to_success_list[penalization_factor] = []

      original_penalization_factor = (
        criteria_dict[contour_type]['positional_penalization']
      )
      criteria_dict[contour_type]['positional_penalization'] = (
        penalization_factor
      )

      for case_info in expected_contour_to_cases[expected_contour_key]:
        target_expected_contour = case_info[0]
        candidate_contours = case_info[1]
        correct_candidate_index = case_info[2]
        case_title = case_info[3]

        scores = []
        for candidate_contour in candidate_contours:
          target_expected_contour.prepare(
            candidate_contour,
            image_width=301,
            image_height=462,
          )
          score = target_expected_contour.shape_restrictions(criteria_dict)
          scores.append(score)
        chosen_candidate_index = int(np.argmin(scores))
        if chosen_candidate_index == correct_candidate_index:
          penalization_to_success_list[penalization_factor].append(True)
        else:
          penalization_to_success_list[penalization_factor].append(False)

      criteria_dict[contour_type]['positional_penalization'] = (
        original_penalization_factor
      )
      
    expected_contour_to_penalization_to_precision = {}
    for penalization_factor in first_stage_penalization_factors:
      positive_amount = penalization_to_success_list[penalization_factor].count(True)
      length = len(penalization_to_success_list[penalization_factor])
      expected_contour_to_penalization_to_precision[penalization_factor] = positive_amount / length

    all_precisions_first_stage = list(expected_contour_to_penalization_to_precision.values())
    best_precision_penalization_factor_index = (
      len(all_precisions_first_stage) - 1 - np.argmax(all_precisions_first_stage[::-1])
    )
    best_precision_penalization_factor = first_stage_penalization_factors[
      best_precision_penalization_factor_index]

    upper_bound = min(1, best_precision_penalization_factor + (step * (range / 2)))
    lower_bound = max(0, best_precision_penalization_factor - (step * (range / 2)))
    second_stage_penalization_factors = (
      list(np.arange(upper_bound, best_precision_penalization_factor, (-1) * step)) +
      list(np.arange(best_precision_penalization_factor, lower_bound, (-1) * step))
    )
    second_stage_penalization_factors = [
      float(factor) for factor in second_stage_penalization_factors
    ]

    penalization_to_success_list = {}
    for penalization_factor in second_stage_penalization_factors:
      penalization_to_success_list[penalization_factor] = []

      original_penalization_factor = (
        criteria_dict[contour_type]['positional_penalization']
      )
      criteria_dict[contour_type]['positional_penalization'] = (
        penalization_factor
      )

      for case_info in expected_contour_to_cases[expected_contour_key]:
        target_expected_contour = case_info[0]
        candidate_contours = case_info[1]
        correct_candidate_index = case_info[2]
        case_title = case_info[3]

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key][case_title][penalization_factor] = {}
        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key][case_title][penalization_factor] = {
            'correct_candidate_index': correct_candidate_index,
            'chosen_candidate_index': None,
            'candidate_contour_differences': [],
          }

        scores = []
        for candidate_contour in candidate_contours:
          target_expected_contour.prepare(
            candidate_contour,
            image_width=301,
            image_height=462,
          )
          score = target_expected_contour.shape_restrictions(criteria_dict)
          scores.append(score)
          expected_contour_to_case_to_contour_to_difference[
            expected_contour_key][case_title][penalization_factor][
              'candidate_contour_differences'].append(score)

        chosen_candidate_index = int(np.argmin(scores))
        if chosen_candidate_index == correct_candidate_index:
          penalization_to_success_list[penalization_factor].append(True)
        else:
          penalization_to_success_list[penalization_factor].append(False)

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title][penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

      criteria_dict[contour_type]['positional_penalization'] = (
        original_penalization_factor
      )

    for penalization_factor in second_stage_penalization_factors:
      positive_amount = penalization_to_success_list[penalization_factor].count(True)
      length = len(penalization_to_success_list[penalization_factor])
      expected_contour_to_factor_to_precision[expected_contour_key][penalization_factor] = (
        positive_amount / length
      )

  # Different expected_contours of the same contour_type may have different factors
  # attempted. Make it homogeneous; same contour_type same penalization factors so
  # that an average precision per factor can be calculated for each contour type.
  for contour_type in criteria_dict.keys():
    relevant_expected_contour_keys = list(filter(
      lambda key : contour_type in key.split('_'),
      expected_contour_to_factor_to_precision.keys(),
    ))

    all_penalization_factor_keys = []
    for relevant_expected_contour_key in relevant_expected_contour_keys:
      all_penalization_factor_keys = all_penalization_factor_keys + (
        list(
          (
            expected_contour_to_factor_to_precision[relevant_expected_contour_key]
          ).keys()
        )
      )

    for relevant_expected_contour_key in relevant_expected_contour_keys:
      penalization_to_success_list = {}
      for penalization_factor in all_penalization_factor_keys:
        if penalization_factor not in expected_contour_to_factor_to_precision[relevant_expected_contour_key]:
          missing_penalization_factor = penalization_factor
          penalization_to_success_list[missing_penalization_factor] = []

          original_penalization_factor = (
            criteria_dict[contour_type]['positional_penalization']
          )
          criteria_dict[contour_type]['positional_penalization'] = (
            missing_penalization_factor
          )
          for case_info in expected_contour_to_cases[relevant_expected_contour_key]:
            target_expected_contour = case_info[0]
            candidate_contours = case_info[1]
            correct_candidate_index = case_info[2]
            case_title = case_info[3]

            expected_contour_to_case_to_contour_to_difference[
                      relevant_expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key][case_title][missing_penalization_factor] = {
                'correct_candidate_index': correct_candidate_index,
                'chosen_candidate_index': None,
                'candidate_contour_differences': [],
              }

            scores = []
            for candidate_contour in candidate_contours:
              target_expected_contour.prepare(
                candidate_contour,
                image_width=301,
                image_height=462,
              )
              score = target_expected_contour.shape_restrictions(criteria_dict)
              scores.append(score)
              expected_contour_to_case_to_contour_to_difference[
                relevant_expected_contour_key][case_title][missing_penalization_factor][
                  'candidate_contour_differences'].append(score)

            chosen_candidate_index = int(np.argmin(scores))

            if chosen_candidate_index == correct_candidate_index:
              penalization_to_success_list[penalization_factor].append(True)
            else:
              penalization_to_success_list[penalization_factor].append(False)

            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key
              ][case_title][missing_penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

          criteria_dict[contour_type]['positional_penalization'] = (
            original_penalization_factor
          )
        
      for penalization_factor in penalization_to_success_list.keys():
        positive_amount = penalization_to_success_list[penalization_factor].count(True)
        length = len(penalization_to_success_list[penalization_factor])
        expected_contour_to_factor_to_precision[relevant_expected_contour_key][penalization_factor] = (
          positive_amount / length
        )

  contour_type_to_factor_to_average_precision = {}
  for contour_type in criteria_dict.keys():
    relevant_expected_contour_keys = list(filter(
      lambda key : contour_type in key.split('_'),
      expected_contour_to_factor_to_precision.keys(),
    ))

    if len(relevant_expected_contour_keys) > 0:
      contour_type_to_factor_to_average_precision[contour_type] = {}
      # penalization factors used are the same (relevant keys wise)
      penalization_factor_keys = (
        expected_contour_to_factor_to_precision[relevant_expected_contour_keys[0]]
      ).keys()
      
      for penalization_factor in penalization_factor_keys:
        relevant_precisions = [
          expected_contour_to_factor_to_precision[relevant_expected_contour_key][penalization_factor]
          for relevant_expected_contour_key in relevant_expected_contour_keys
        ]
        average_precision = sum(relevant_precisions) / len(relevant_precisions)
        contour_type_to_factor_to_average_precision[contour_type][penalization_factor] = average_precision

  contour_type_to_best_factor = {}
  for contour_type in criteria_dict.keys():
    if contour_type in contour_type_to_factor_to_average_precision:
      local_average_precisions = list(
        contour_type_to_factor_to_average_precision[contour_type].values()
      )
      best_factor_index = (
        len(local_average_precisions) - 1 - np.argmax(local_average_precisions[::-1])
      )

      contour_type_to_best_factor[contour_type] = {
        'best_factor': (
          list(
            (
              contour_type_to_factor_to_average_precision[contour_type]
            ).keys()
          )[best_factor_index]
        ),
        'precision': local_average_precisions[best_factor_index]
      }
    else:
      contour_type_to_best_factor[contour_type] = 'No se encontraron casos.'

  tables_generated = 1
  table_cut_count = 0
  TABLE_CUT_THRESHOLD = 12

  for contour_type_key in contour_type_to_best_factor:
    processed_contour_type_key = contour_type_key.replace('_', '\\_')
    if isinstance(contour_type_to_best_factor[contour_type_key], dict):
      best_factor = contour_type_to_best_factor[contour_type_key]['best_factor']
      precision = contour_type_to_best_factor[contour_type_key]['precision']
      table_entry = f'{processed_contour_type_key} & {best_factor} & {precision}' + (
        f' \\\\ \n')

      table_cut_count += 1
      if table_cut_count >= TABLE_CUT_THRESHOLD:
        tables_generated += 1
        output_string = output_string + cut_table(table_entry,
                                                  encounter_amount=tables_generated)
        table_cut_count = 0
      else:
        output_string = output_string + table_entry
      output_string = output_string + '\\hline\n'
    else:
      feedback_str = contour_type_to_best_factor[contour_type_key]
      table_entry = f'{processed_contour_type_key} & {feedback_str} & -' + (
        f' \\\\ \n')

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

  with open('tab_best_factor.txt', 'w', encoding='utf-8') as f:
    f.write(output_string)
    print('Writing tab_best_factor.txt')
    print('Success.')

