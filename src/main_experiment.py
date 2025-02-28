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

def get_fit_dictionary(images: list, selected_group: Union[str, list]) -> dict:
  fit = {}
  ages = {}

  if isinstance(selected_group, str):
    reference_measurements = BONE_AGE_ATLAS[selected_group]
  if isinstance(selected_group, list):
    reference_measurements = [
      BONE_AGE_ATLAS[target_age] for target_age in selected_group
    ]

  for i, image_info in enumerate(images):
    filename = image_info[0]
    image = image_info[1]

    (
      estimated_age,
      measurements,
      image_stage_1,
      image_stage_2,
    ) = estimate_age_from_image(image, nofilter=True) # TODO turn this off after testing the experiment

    if estimated_age != -1 and estimated_age != -2:
      for measurement_key in measurements:
        measurement_value = measurements[measurement_key]

        if measurement_key not in fit:
          fit[measurement_key] = 0
        else:
          if isinstance(reference_measurements, dict):
            error = (
              (
                reference_measurements[measurement_key] - measurement_value
              ) ** 2
            )
          elif isinstance(reference_measurements, list):
            error = (
              (
                reference_measurements[i][measurement_key] - measurement_value
              ) ** 2
            )
            
          fit[measurement_key] = fit[measurement_key] + error
        
        ages[filename] = estimated_age
    elif estimated_age == -1:
      raise ValueError(f'index={i} radiography\'s estimation failed.' \
                        ' Failed sesamoid search.')
    elif estimated_age == -2:
      raise ValueError(f'index={i} radiography\'s estimation failed.' \
                        ' Failed fingers search.')

  for measurement_key in fit:
    fit[measurement_key] = fit[measurement_key] / len(fit[measurement_key])

  return fit, ages

def experiment(
    images: Union[dict, list],
    selected_group: str = None,
    control_dict: dict = None
):
  if isinstance(images, list):
    if selected_group == 'control':
      filenames = images[:, 0]
      ages = [control_dict[filename] for filename in filenames]
      selected_group = ages
      fit, ages = get_fit_dictionary(images, selected_group)
    else:
      fit, ages = get_fit_dictionary(images, selected_group)

    print(f'Fit results for group {selected_group}:')
    for measurement_key in fit:
      fit[measurement_key] = fit[measurement_key] / len(fit[measurement_key])
      print(f'{measurement_key}={fit[measurement_key]}')
    print('\n')

    print('Ages:')
    if selected_group == 'control':
      for filename in control_dict:
        expected_age = control_dict[filename]
        estimated_age = ages[filename]
        print(f'{filename} expected age={expected_age}, ' \
              f'estimated age={estimated_age}')
      print('\n')
    else:
      for filename in ages:
        estimated_age = ages[filename]
        print(f'{filename} expected age={selected_group}, ' \
              f'estimated age={estimated_age}')
      print('\n')
  elif isinstance(images, dict):
    for images_key in images:
      if images_key == 'group_17_5':
        selected_group = '17.5'
      elif images_key == 'group_18_5':
        selected_group = '18.5'
      elif images_key == 'group_19_5':
        selected_group = '19.5'
      elif images_key == 'group_control':
        images_content = images[images_key]
        filenames = images_content[:, 0]
        ages = [control_dict[filename] for filename in filenames]
        selected_group = ages

      fit, ages = get_fit_dictionary(images[images_key], selected_group)

      print(f'Fit results for group {selected_group}:')
      for measurement_key in fit:
        fit[measurement_key] = fit[measurement_key] / len(fit[measurement_key])
        print(f'{measurement_key}={fit[measurement_key]}')
      print('\n')

      if selected_group == 'control':
        for filename in control_dict:
          expected_age = control_dict[filename]
          estimated_age = ages[filename]
          print(f'{filename} expected age={expected_age}, ' \
                f'estimated age={estimated_age}')
        print('\n')
      else:
        print(f'Ages for group {selected_group}:')
        for filename in ages:
          estimated_age = ages[filename]
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
    groupcontrol: str
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
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            images.append([image_path.name, image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")
    
    control_dict = None
    if selected_group == 'control':
      # Read JSON file with filename-age information
      try:
        with open(groupcontrol, 'r') as file:
          image_age_dict = json.load(file)

          if not isinstance(image_age_dict, dict):
            raise ValueError(f"Error: JSON file does not contain a dictionary ' \
                             'at the top level.")
          for filename, age in image_age_dict.items():
            if not isinstance(filename, str):
              raise ValueError(f"Error: JSON dictionary keys should be' \
                               ' filenames (strings). Found key: {filename} ' \
                                'which is not a string.")
            if not isinstance(age, str):
              raise ValueError(f"Error: JSON dictionary values should be age ' \
                                '(string). Value for '{filename}' is not a ' \
                                'string: {age}")
            
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
                         f"entries. Expected {len(images)} but got" \
                           f" {len(control_dict)}")

    experiment(images, selected_group, control_dict)
  else:
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

    image_dir_path_18_5 = Path(group17_5)
    if not image_dir_path_18_5.is_dir():
      print(f'Error: {group18_5} is not a valid directory.')
      return

    image_dir_path_19_5 = Path(group17_5)
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
            images['group_17_5'].append([image_path.name, image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_18_5.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            images['group_18_5'].append([image_path.name, image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_19_5.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            images['group_19_5'].append([image_path.name, image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    for image_path in image_dir_path_control.glob('*'):
      if image_path.is_file() and image_path.suffix.lower() in image_formats:
        try:
          with Image.open(image_path) as image:
            print(f"Processing image: {image_path.name}")
            images['group_control'].append([image_path.name, image])
        except Exception as e:
          print(f"Error opening image {image_path.name}: {e}")

    # Read JSON file with filename-age information for control group
    control_dict = None
    try:
      with open(groupcontrol, 'r') as file:
        image_age_dict = json.load(file)

        if not isinstance(image_age_dict, dict):
          raise ValueError(f"Error: JSON file does not contain a dictionary ' \
                            'at the top level.")
        for filename, age in image_age_dict.items():
          if not isinstance(filename, str):
            raise ValueError(f"Error: JSON dictionary keys should be' \
                              ' filenames (strings). Found key: {filename} ' \
                              'which is not a string.")
          if not isinstance(age, str):
            raise ValueError(f"Error: JSON dictionary values should be age ' \
                              '(string). Value for '{filename}' is not a ' \
                              'string: {age}")
          
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
                       f" Expected {len(images['group_control'])} but got " \
                        f" {len(control_dict)}")

    experiment(images, None, control_dict)
