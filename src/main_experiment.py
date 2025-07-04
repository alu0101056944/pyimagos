'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Experiment of age estimation for four different groups:

- A group of radiographies aged 17.5
- A group of radiographies aged 18.5
- A group of radiographies aged 19.5
- A control group

Where each group has a fit for a given set of measurements (see constants.py)

Prints tables with the information through the terminal.
'''

from pathlib import Path
from PIL import Image
from typing import Union
import json

import numpy as np

from src.main_execute import estimate_age_from_image
from constants import BONE_AGE_ATLAS

def get_fit_dictionary(images: list, selected_group: Union[float, None],
                       nofilter: bool = False, use_cpu: bool = True,
                       noresize: bool = False, useinput2: bool = False,
                       start_indices_dict: dict = None,
                       debug_mode: bool = False) -> dict:
  fit = {}
  ages = {}
  filename_to_measurements = {}
  output_case_to_contours_sequence = {
    
  }

  if isinstance(selected_group, float):
    reference_measurements = BONE_AGE_ATLAS[str(selected_group)]

    for i, image_info in enumerate(images):
      filename = image_info[0]
      image = image_info[1]
      start_index = -1
      if start_indices_dict is not None:
        start_index = start_indices_dict[filename]

      print(f'\nExperiment execution with {filename}.')

      if not useinput2:
        if not debug_mode:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize, start_index=start_index,
                                      debug_mode=debug_mode)
        else:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
            contours_sequence
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize, start_index=start_index,
                                      debug_mode=debug_mode)
          output_case_to_contours_sequence[filename] = contours_sequence
      else:
        image_stage_2_str = image_info[2]
        image_stage_2 = None
        try:
          with Image.open(image_stage_2_str) as image2:
            if image2.mode == 'L':
              image2 = image2.convert('RGB')
              image_stage_2 = np.array(image2)
            elif image2.mode == 'RGB':
              image_stage_2 = np.array(image2)
        except Exception as e:
          print(f"Error opening image {image_stage_2}: {e}")
          raise

        if not debug_mode:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      input_image_2=image_stage_2,
                                      start_index=start_index,
                                      debug_mode=debug_mode)
        else:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
            contours_sequence
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      input_image_2=image_stage_2,
                                      start_index=start_index,
                                      debug_mode=debug_mode)
          output_case_to_contours_sequence[filename] = contours_sequence

      if filename not in filename_to_measurements:
        filename_to_measurements[filename] = {}

      if estimated_age != -1 and estimated_age != -2:
        for measurement_key in measurements:
          measurement_value = measurements[measurement_key]

          if measurement_key in reference_measurements:
            error = (
              (
                reference_measurements[measurement_key] - measurement_value
              ) ** 2
            )
            if measurement_key in fit:
              fit[measurement_key] = fit[measurement_key] + error
            else:
              fit[measurement_key] = error
            
            filename_to_measurements[filename][measurement_key] = (
              measurement_value
            )
          
          ages[filename] = {
            'estimated': estimated_age,
            'real': selected_group
          }

      elif estimated_age == -1:
        ages[filename] = {
          'estimated': -1,
          'real': selected_group
        }
        filename_to_measurements[filename] = None
      elif estimated_age == -2:
        ages[filename] = {
          'estimated': -2,
          'real': selected_group
        }
        filename_to_measurements[filename] = None
  elif selected_group is None:
    for i, image_info in enumerate(images):
      filename = image_info[0]
      image = image_info[1]

      if not useinput2:
        if not debug_mode:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      debug_mode=debug_mode)
        else:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
            contours_sequence,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      debug_mode=debug_mode)
          output_case_to_contours_sequence[filename] = contours_sequence
      else:
        image_stage_2_str = image_info[2]
        image_stage_2 = None
        try:
          with Image.open(image_stage_2_str) as image2:
            if image2.mode == 'L':
              image2 = image2.convert('RGB')
              image_stage_2 = np.array(image2)
            elif image2.mode == 'RGB':
              image_stage_2 = np.array(image2)
        except Exception as e:
          print(f"Error opening image {image_stage_2}: {e}")
          raise

        if not debug_mode:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      input_image_2=image_stage_2,
                                      debug_mode=debug_mode)
        else:
          (
            estimated_age,
            measurements,
            image_stage_1,
            image_stage_2,
            contours_sequence,
          ) = estimate_age_from_image(image, nofilter=nofilter, use_cpu=use_cpu,
                                      noresize=noresize,
                                      input_image_2=image_stage_2,
                                      debug_mode=debug_mode)
          output_case_to_contours_sequence[filename] = contours_sequence

      if estimated_age != -1 and estimated_age != -2:
        smallest_difference = -1
        closest_reference_measurements_key = None
        for reference_measurements_key in BONE_AGE_ATLAS:
          reference_measurements = BONE_AGE_ATLAS[reference_measurements_key]

          difference = 0
          for measurement_key in measurements:
            if measurement_key in reference_measurements:
              difference = difference + abs((
                measurements[measurement_key] - (
                  reference_measurements[measurement_key]
                )
              ))

          if smallest_difference == -1 or difference < smallest_difference:
            closest_reference_measurements_key = reference_measurements_key
            smallest_difference = difference

        reference_measurements = (
          BONE_AGE_ATLAS[closest_reference_measurements_key]
        )
        for measurement_key in measurements:
          measurement_value = measurements[measurement_key]

          if measurement_key in reference_measurements:
            error = (
              (
                reference_measurements[measurement_key] - measurement_value
              ) ** 2
            )
            if measurement_key in fit:
              fit[measurement_key] = fit[measurement_key] + error
            else:
              fit[measurement_key] = error
          
          ages[filename] = {
            'estimated': estimated_age,
            'real': selected_group
          }
      elif estimated_age == -1:
        raise ValueError(f'index={i} radiography\'s estimation failed.' \
                          ' Failed sesamoid search.')
      elif estimated_age == -2:
        raise ValueError(f'index={i} radiography\'s estimation failed.' \
                            ' Failed fingers search.')

  for measurement_key in fit:
    fit[measurement_key] = fit[measurement_key] / len(images)

  if not debug_mode:
    return fit, ages, filename_to_measurements
  else:
    return fit, ages, filename_to_measurements, output_case_to_contours_sequence


def experiment(
    images: Union[dict, list],
    selected_group: str = None,
    control_dict: dict = None,
    nofilter: bool = False,
    use_cpu: bool = True,
    noresize: bool = False,
    useinput2: bool = False,
    start_indices_dict: dict = None,
    debug_mode: bool = False,
):
  if isinstance(images, list):
    if selected_group == 'control':
      if not debug_mode:
        (
          fit,
          ages,
          filename_to_measurements,
        ) = get_fit_dictionary(images, None, nofilter, use_cpu, noresize, useinput2,
                              start_indices_dict, debug_mode)
      else:
        (
          fit,
          ages,
          filename_to_measurements,
          case_to_contours_sequence,
        ) = get_fit_dictionary(images, None, nofilter, use_cpu, noresize, useinput2,
                              start_indices_dict, debug_mode)
        return fit, ages, filename_to_measurements, case_to_contours_sequence
    else:
      if not debug_mode:
        (
          fit,
          ages,
          filename_to_measurements,
        ) = get_fit_dictionary(images, float(selected_group), nofilter, use_cpu,
                              noresize, useinput2, start_indices_dict, debug_mode)
      else:
        (
          fit,
          ages,
          filename_to_measurements,
          case_to_contours_sequence,
        ) = get_fit_dictionary(images, float(selected_group), nofilter, use_cpu,
                              noresize, useinput2, start_indices_dict, debug_mode)
        return fit, ages, filename_to_measurements, case_to_contours_sequence


    print('\n')
    print(f'Fit results for group {selected_group}:')
    for measurement_key in fit:
      print(f'{measurement_key}={fit[measurement_key]}')
    print('\n')

    if selected_group == 'control':
      print(f'Ages for control group:')
      for filename in control_dict:
        expected_age = control_dict[filename]
        estimated_age = ages[filename]['estimated']
        print(f'{filename} expected age={expected_age}, ' \
              f'estimated age={estimated_age}')
      print('\n')
    else:
      print(f'Ages for group {selected_group}:')
      for filename in ages:
        estimated_age = ages[filename]['estimated']
        print(f'{filename} | expected age={selected_group} | ' \
              f'estimated age={estimated_age}')

      print('\n')

      print(f'Measurements for group {selected_group}')
      for filename in filename_to_measurements:
        measurements = filename_to_measurements[filename]
        if measurements is not None:
          print(f'{filename} | {measurements}')
        else:
          print(f'{filename} | None')

      print('\n')

  elif isinstance(images, dict):
    for images_key in images:
      if images_key == 'group_17_5':
        selected_group = 17.5
      elif images_key == 'group_18_5':
        selected_group = 18.5
      elif images_key == 'group_19_5':
        selected_group = 19.5
      elif images_key == 'group_control':
        selected_group = None

      if not debug_mode:
        fit, ages, _ = get_fit_dictionary(images[images_key], selected_group,
                                          nofilter, use_cpu, noresize, useinput2,
                                          start_indices_dict, debug_mode)
      else:
        fit, ages, _, case_to_contours_sequence = get_fit_dictionary(
          images[images_key],
          selected_group,
          nofilter,
          use_cpu,
          noresize,
          useinput2,
          start_indices_dict,
          debug_mode
        )
        return fit, ages, _, case_to_contours_sequence
      print(f'Fit results for group {selected_group}:')
      for measurement_key in fit:
        print(f'{measurement_key}={fit[measurement_key]}')
      print('\n')

      if selected_group == None:
        print(f'Ages for control group:')
        for filename in control_dict:
          expected_age = control_dict[filename]
          estimated_age = ages[filename]['estimated']
          print(f'{filename} expected age={expected_age}, ' \
                f'estimated age={estimated_age}')
        print('\n')
      else:
        print(f'Ages for group {selected_group}:')
        for filename in ages:
          estimated_age = ages[filename]['estimated']
          print(f'{filename} expected age={selected_group}, ' \
                f'estimated age={estimated_age}')
        print('\n')
  else:
    raise ValueError('Experiment expected a dict or a list and got neither.')

def main_experiment(
    single: bool,
    group17_5: str,
    group18_5: str,
    group19_5: str,
    groupcontrol: str,
    nofilter: bool = False,
    use_cpu: bool = True,
    noresize: bool = False,
    useinput2: bool = False,
    use_starts_file: bool = False,
    debug_mode: bool = False,
):
  if single:
    amount_of_passed_options = (
      (np.array([group17_5, group18_5, group19_5, groupcontrol]) != None).sum()
    )
    if amount_of_passed_options != 1:
      print(f'Error: passed {amount_of_passed_options} groupX options ' \
                 'but expected only 1.')
      return
  
    non_none_path = None
    non_none_path_string = None
    selected_group = None
    if group17_5 is not None:
      non_none_path = group17_5
      non_none_path_string = group17_5
      selected_group = '17.5'
    elif group18_5 is not None:
      non_none_path = group18_5
      non_none_path_string = group18_5
      selected_group = '18.5'
    elif group19_5 is not None:
      non_none_path = group19_5
      non_none_path_string = group19_5
      selected_group = '19.5'
    elif groupcontrol is not None:
      non_none_path = groupcontrol
      non_none_path_string = groupcontrol
      selected_group = 'control'

    non_none_path = Path(non_none_path)

    if not non_none_path.is_dir():
      print(f'Error: {non_none_path_string} is not a valid directory.',
                 err=True)
      return

    images = []

    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    for image_path in non_none_path.glob('*'):
      if (image_path.is_file() and image_path.suffix.lower() in image_formats and
          '_stage_2' not in image_path.stem):
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            if image.mode == 'L':
              image = image.convert('RGB')
              numpy_image = np.array(image)
            elif image.mode == 'RGB':
              numpy_image = np.array(image)
            if useinput2:
              images.append(
                [
                  image_path.name,
                  numpy_image,
                  f'{image_path.absolute().parent / image_path.stem}' \
                    f'_stage_2{image_path.suffix}'
                ]
              )
            else:
              images.append([image_path.name, numpy_image])

        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    start_indices_dict = None
    if use_starts_file:
      image_names = [image_content[0] for image_content in images]
      for file_path in non_none_path.glob('*'):
        if (file_path.is_file() and file_path.stem.lower() == 'start_indices'):
          # Read JSON file with filename-start contour index information
          try:
            with open(file_path, 'r') as file:
              filename_to_start_index_dict = json.load(file)

              if not isinstance(filename_to_start_index_dict, dict):
                raise ValueError(f"Error: start_indices.json JSON file does not ' \
                                 'contain a dictionary at the top level.")
              for filename, start_index in filename_to_start_index_dict.items():
                if not isinstance(filename, str):
                  raise ValueError(f"Error: JSON dictionary keys should be' \
                                  ' filenames (strings). Found key: {filename} ' \
                                    'which is not a string.")
                if not isinstance(start_index, (int, int)):
                  raise ValueError(f"Error: JSON dictionary values should be index ' \
                                    '(int). Value for '{filename}' is not an ' \
                                    '(int): {start_index}")
                if not filename in image_names:
                  raise ValueError(f"Error: Filename key {filename} in the JSON " \
                                   "is not an actual file")
                
              start_indices_dict = filename_to_start_index_dict
          except FileNotFoundError:
            print(f"Error: start_indices JSON file not found at path: {groupcontrol}")
            return None
          except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON from file start_indices.json. Invalid ' \
                  'JSON format.\nDetails: {e}")
            return None
          except Exception as e:
            print(f"An unexpected error occurred while reading the JSON file: {e}")
            return None
          
          if len(start_indices_dict) != len(images):
            raise ValueError(f"Error: start_indices.json file is missing " \
                            f"entries. Expected {len(images)} but found" \
                              f" {len(start_indices_dict)}")
    
    # TODO --single mode has received changes to non control groups, so control
    # group will probably raise error. Fix it.

    control_dict = None
    if selected_group == 'control':
      control_image_names = [image_content[0] for image_content in images]
      for file_path in non_none_path.glob('*'):
        if (file_path.is_file() and file_path.suffix.lower() == '.json'):
          # Read JSON file with filename-age information
          try:
            with open(file_path, 'r') as file:
              image_age_dict = json.load(file)

              if not isinstance(image_age_dict, dict):
                raise ValueError(f"Error: JSON file does not contain a dictionary ' \
                                'at the top level.")
              for filename, age in image_age_dict.items():
                if not isinstance(filename, str):
                  raise ValueError(f"Error: JSON dictionary keys should be' \
                                  ' filenames (strings). Found key: {filename} ' \
                                    'which is not a string.")
                if not isinstance(age, (int, float)):
                  raise ValueError(f"Error: JSON dictionary values should be age ' \
                                    '(int, float). Value for '{filename}' is not a ' \
                                    '(int, float): {age}")
                if not filename in control_image_names:
                  raise ValueError(f"Error: Filename key {filename} in the JSON " \
                                   "is not an actual file")
                
              control_dict = image_age_dict
          except FileNotFoundError:
            print(f"Error: JSON file not found at path: {groupcontrol}")
            return None
          except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON from file: {groupcontrol}. Invalid ' \
                  'JSON format.\nDetails: {e}")
            return None
          except Exception as e:
            print(f"An unexpected error occurred while reading the JSON file: {e}")
            return None
          
          if len(control_dict) != len(images):
            raise ValueError(f"Error: control group's json file is missing " \
                            f"entries. Expected {len(images)} but found" \
                              f" {len(control_dict)}")
          
    if not debug_mode:
      experiment(images, selected_group, control_dict, nofilter, use_cpu,
                noresize, useinput2, start_indices_dict)
    else:
      return experiment(images, selected_group, control_dict, nofilter, use_cpu,
                noresize, useinput2, start_indices_dict, debug_mode)

  else:
    # TODO --single mode has received changes to non control groups, so not --single
    # group is not updated and may raise errors. Fix it.

    if group17_5 is None:
      print(f'Error: missing group17_5 option.')
      return
    elif group18_5 is None:
      print(f'Error: missing group18_5 option.')
      return
    elif group19_5 is None:
      print(f'Error: missing group19_5 option.')
      return
    elif groupcontrol is None:
      print(f'Error: missing groupcontrol option.')
      return
    
    image_dir_path_17_5 = Path(group17_5)
    if not image_dir_path_17_5.is_dir():
      print(f'Error: {group17_5} is not a valid directory.')
      return

    image_dir_path_18_5 = Path(group18_5)
    if not image_dir_path_18_5.is_dir():
      print(f'Error: {group18_5} is not a valid directory.')
      return

    image_dir_path_19_5 = Path(group19_5)
    if not image_dir_path_19_5.is_dir():
      print(f'Error: {group19_5} is not a valid directory.')
      return
    
    image_dir_path_control = Path(groupcontrol)
    if not image_dir_path_control.is_dir():
      print(f'Error: {groupcontrol} is not a valid directory.')
      return
    
    images = {
      'group_17_5': [],
      'group_18_5': [],
      'group_19_5': [],
      'group_control': [],
    }

    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    for image_path in image_dir_path_17_5.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            if image.mode == 'L':
              image = image.convert('RGB')
              numpy_image = np.array(image)
            elif image.mode == 'RGB':
              numpy_image = np.array(image)
            if useinput2:
              images['group_17_5'].append(
                [
                  image_path.name,
                  numpy_image,
                  f'{image_path.absolute().parent / image_path.stem}' \
                    f'_stage_2{image_path.suffix}'
                ]
              )
            else:
              images['group_17_5'].append([image_path.name, numpy_image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_18_5.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            if image.mode == 'L':
              image = image.convert('RGB')
              numpy_image = np.array(image)
            elif image.mode == 'RGB':
              numpy_image = np.array(image)
            if useinput2:
              images['group_18_5'].append(
                [
                  image_path.name,
                  numpy_image,
                  f'{image_path.absolute().parent / image_path.stem}' \
                    f'_stage_2{image_path.suffix}'
                ]
              )
            else:
              images['group_18_5'].append([image_path.name, numpy_image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_19_5.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            if image.mode == 'L':
              image = image.convert('RGB')
              numpy_image = np.array(image)
            elif image.mode == 'RGB':
              numpy_image = np.array(image)
            if useinput2:
              images['group_19_5'].append(
                [
                  image_path.name,
                  numpy_image,
                  f'{image_path.absolute().parent / image_path.stem}' \
                    f'_stage_2{image_path.suffix}'
                ]
              )
            else:
              images['group_19_5'].append([image_path.name, numpy_image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_control.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            if image.mode == 'L':
              image = image.convert('RGB')
              numpy_image = np.array(image)
            elif image.mode == 'RGB':
              numpy_image = np.array(image)
            if useinput2:
              images['group_control'].append(
                [
                  image_path.name,
                  numpy_image,
                  f'{image_path.absolute().parent / image_path.stem}' \
                    f'_stage_2{image_path.suffix}'
                ]
              )
            else:
              images['group_control'].append([image_path.name, numpy_image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    # Read JSON file with filename-age information for control group
    control_dict = None
    control_image_names = [
      image_content[0] for image_content in images['group_control']
    ]
    for file_path in image_dir_path_control.glob('*'):
      if (file_path.is_file() and file_path.suffix.lower() == '.json'):
        # Read JSON file with filename-age information
        try:
          with open(file_path, 'r') as file:
            image_age_dict = json.load(file)

            if not isinstance(image_age_dict, dict):
              raise ValueError(f"Error: JSON file does not contain a dictionary ' \
                                'at the top level.")
            for filename, age in image_age_dict.items():
              if not isinstance(filename, str):
                raise ValueError(f"Error: JSON dictionary keys should be' \
                                  ' filenames (strings). Found key: {filename} ' \
                                  'which is not a string.")
              if not isinstance(age, (int, float)):
                raise ValueError(f"Error: JSON dictionary values should be age ' \
                                  '(int, float). Value for '{filename}' is not a ' \
                                  '(int, float): {age}")
            
              if not filename in control_image_names:
                      raise ValueError(f"Error: Filename key {filename} in the JSON " \
                                      "is not an actual file")
              
            control_dict = image_age_dict
        except FileNotFoundError:
          print(f"Error: JSON file not found at path: {groupcontrol}")
          raise
        except json.JSONDecodeError as e:
          print(f"Error: Could not parse JSON from file: {groupcontrol}. Invalid ' \
                'JSON format.\nDetails: {e}")
          raise
        except Exception as e:
          print(f"An unexpected error occurred while reading the JSON file: {e}")
          raise
        
        if len(control_dict) != len(images['group_control']):
          raise ValueError(f"Error: control group's json file is missing entries. " \
                          f" Expected {len(images['group_control'])} but found " \
                            f" {len(control_dict)}")
        
    print('\n')

    experiment(images, None, control_dict, nofilter, use_cpu, noresize,
               useinput2)
