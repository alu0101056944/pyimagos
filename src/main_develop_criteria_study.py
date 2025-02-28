'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Test criteria parameter variations age estiamtion results.
'''

import os.path
import itertools
import copy

from PIL import Image
import numpy as np

from src.main_execute import estimate_age_from_image
from constants import CRITERIA_DICT, CRITERIA_DICT_VARIATION_MAGNITUDES

def generate_nested_variations(original_params, x_dict):
  """
  Returns:
      list: List of dictionaries with all possible ±x variations.
  """
  assert validate_structure(original_params, x_dict),  \
    "original_params and x_dict must have the same structure"

  # Flatten the parameters and x-values into a list of (path, original_value, x_value)
  flattened = list(flatten_with_x(original_params, x_dict))
  
  # Generate all possible combinations of signs (+1, -1) for each parameter
  num_params = len(flattened)
  sign_combinations = itertools.product([1, -1], repeat=num_params)

  variations = []
  for j, signs in enumerate(sign_combinations):
    variation = copy.deepcopy(original_params)

    # TODO: remove this limit after testing
    if j > 10:
      break

    for i, (path, original_val, x_val) in enumerate(flattened):
      print(f'Variation generated ({j}, {i})')

      # TODO: remove this limit after testing
      if i > 50:
        break

      sign = signs[i]
      keys = path.split('.')
      current_dict = variation

      # Navigate to the correct nested dictionary
      for key in keys[:-1]:
        current_dict = current_dict[key]
      # Update the value with the signed x variation
      current_dict[keys[-1]] = original_val + sign * x_val
    variations.append(variation)
  return variations

def validate_structure(a, b):
  if isinstance(a, dict) and isinstance(b, dict):
    if a.keys() != b.keys():
      return False
    for key in a:
      if not validate_structure(a[key], b[key]):
        return False
    return True
  elif not isinstance(a, dict) and not isinstance(b, dict):
    return True
  else:
    return False
  
def flatten_with_x(params, x_dict, parent_key='', sep='.'):
  """
  Flattens a nested dictionary into a list of (path, value, x_value).
  """
  items = []
  for key in params:
    current_key = f"{parent_key}{sep}{key}" if parent_key else key
    if isinstance(params[key], dict):
      items.extend(flatten_with_x(params[key], x_dict[key], current_key, sep))
    else: 
      items.append((current_key, params[key], x_dict[key])) 
  return items

def test_criteria_parameters(filename: str, outputfilename: str,
                             nofilter: bool = False):
  input_image = None
  try:
    with Image.open(filename) as image:
      input_image = np.array(image)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  print('Variation calculations start.')

  output_string = ''
  variation_dicts = generate_nested_variations(
    CRITERIA_DICT,
    CRITERIA_DICT_VARIATION_MAGNITUDES
  )

  for variation_dict in variation_dicts:
    (
      estimated_age,
      measurements,
      image_stage_1,
      image_stage_2,
    ) = estimate_age_from_image(image, nofilter=nofilter, full_silent=True)
    string = f'{estimated_age}, {variation_dict}'
    output_string += string + "\n"

  print(f'Writing results to {outputfilename}')
  base_filename = os.path.splitext(os.path.basename(outputfilename))[0]
  output_filename = "criteria_variation_report.txt"
  with open(output_filename, 'w') as f:
    f.write(output_string)
  print(f"Results written to: {output_filename}")
