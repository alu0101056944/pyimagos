'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Studying a good image target size so that the contrast enhancement transformer
does not take too long to output the attention map.
'''

from PIL import Image
import time

import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from src.image_filters.contrast_enhancement import ContrastEnhancement

def canny(image: np.array, lower_thresh: str, higher_thresh: str):
  '''Process the radiography image file to border detect it.'''

  if lower_thresh is None or higher_thresh is None:
    raise ValueError('Missing either lower_threshold or higher_threshold option')

  input_image = transforms.ToTensor()(image)
  he_enchanced = ContrastEnhancement().process(input_image)
  he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX,
                              cv.CV_8U)

  gaussian_blurred = cv.GaussianBlur(he_enchanced, (3, 3), 0)

  borders_detected = cv.Canny(gaussian_blurred, float(lower_thresh),
                              float(higher_thresh))
  borders_detected = cv.normalize(borders_detected, None, 0, 255,
                                    cv.NORM_MINMAX, cv.CV_8U)
  
def execute_resize_study():
  size_to_time = {}
  sizes = [[480, 360], [640, 480], [1280, 720]]
  for size in sizes:
    width = size[0]
    height = size[1]

    input_image = None
    filename = 'docs/radiography.jpg'
    try:
      with Image.open(filename) as image:
        input_image = np.array(image)
    except Exception as e:
      print(f"Error opening image {filename}: {e}")
      raise

    scaleFactorX = width / input_image.shape[1]
    scaleFactorY = height / input_image.shape[0]
    resized_image = cv.resize(
      input_image,
      (0, 0),
      fx=scaleFactorX,
      fy=scaleFactorY
    )
    
    start_time = time.time()
    canny(resized_image, 40, 135)
    elapsed_time = time.time() - start_time

    size_to_time[f'{width}x{height}'] = elapsed_time

  print('size, elapsed_time')
  for size, elapsed_time in size_to_time.items():
    print(f'{size}, {elapsed_time}')
  print('\n')
  
