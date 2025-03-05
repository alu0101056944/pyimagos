'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Write image compositions to be able to visualize the best combination of
filters on a set of radiographies.
'''

import os.path
from PIL import Image

from src.image_filters.contrast_enhancement import ContrastEnhancement

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

def make_composition(filename):
  image = None
  try:
    with Image.open(filename) as imagefile:
      image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  scales = [1, 0.6, 0.5, 0.4, 0.3]
  min_thresholds = [5, 10, 15, 20, 25, 30]
  
  for i in range(2):
    if i == 0:
      he_image = transforms.ToTensor()(np.copy(image))
      he_image = ContrastEnhancement(use_cpu=True, noresize=False).process(image)
      he_image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    else:
      he_image = image

    for j in range(2):
      if j == 0:
        gaussian_image = cv.GaussianBlur(he_image, (3, 3), 0)
      else:
        gaussian_image = he_image

      for scale in scales:
        scaled_image = cv.resize(
          gaussian_image,
          (0, 0),
          fx=scale,
          fy=scale,
          interpolation=cv.INTER_AREA
        )

        for min_thresh in min_thresholds:
          canny_image = cv.Canny(scaled_image, min_thresh, 135)
          canny_image = cv.normalize(canny_image, None, 0, 255, cv.NORM_MINMAX,
                                     cv.CV_8U)

          output_filename_string = (
            f'docs/local_images/{os.path.basename(filename)}' \
            f'_canny_study' \
            f'_{'he_on' if i == 0 else 'he_off'}' \
            f'_{'gaussian_on' if j == 0 else 'gaussian_off'}' \
            f'_scale={scale}' \
            f'_min_thresh={min_thresh}' \
            f'.jpg'
          )
          
          cv.imwrite(output_filename_string, canny_image)
          print(f'Wrote image {output_filename_string}')
