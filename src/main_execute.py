'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Processing steps of the radiograph
'''

import os.path
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from src.image_filters.contrast_enhancement import ContrastEnhancement

def process_radiograph(filename: str, write_images: bool = False,
                       show_images: bool = True) -> None:
  input_image = Image.open(filename)
  input_image = transforms.ToTensor()(input_image)

  output_image = ContrastEnhancement().process(input_image)
  output_image = cv.normalize(output_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  ## Border detection
  gaussian_blurred = cv.GaussianBlur(output_image, (5, 5), 0)
  borders_detected = cv.Canny(gaussian_blurred, 40, 135)

  if write_images:
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
               f'processed_canny_borders.jpg',
               borders_detected)
  else:
    if show_images:
      cv.imshow(f'{os.path.basename(filename)}' \
                f'processed_canny_borders.jpg',
                borders_detected)
      cv.waitKey(0)
      cv.destroyAllWindows()
