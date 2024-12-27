'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Output different intermediate results to study alternatives.
'''

import os.path
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.image_filters.border_detection_statistical_range import (
  BorderDetectionStatisticalRange
)

def calculate_watershed(borderImage: np.array) -> np.array:
  numLabels, markers = cv.connectedComponents(borderImage)

  distanceTransform = cv.distanceTransform(borderImage, cv.DIST_L2, 5)
  distanceTransform = cv.normalize(distanceTransform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

  markers = markers.astype(np.int32)
  distanceTransform3Channel = cv.merge((distanceTransform, distanceTransform, distanceTransform))
  watershedImage = cv.watershed(distanceTransform3Channel, markers)
  watershedImage = watershedImage.astype(np.uint8)

  coloredImage = np.zeros((borderImage.shape[0], borderImage.shape[1], 3), dtype=np.uint8)
  uniqueLabels = np.unique(watershedImage)
  for label in uniqueLabels:
    if label == 0:
        continue
    
    mask = watershedImage == label
    b, g, r = np.random.randint(0, 256, 3)
    coloredImage[mask] = (b, g, r)

  return coloredImage

def border_filter_alternatives_analysis(filename: str,
                                        write_images: bool = False,
                                        show_images: bool = True) -> None:
  input_image = Image.open(filename)
  input_image = transforms.ToTensor()(input_image)

  output_image = ContrastEnhancement().process(input_image)
  output_image = cv.normalize(output_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  kernel_sizes = [3, 5, 7, 9]
  gaussian_sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  for kernel_size in kernel_sizes:
    for gaussian_sigma in gaussian_sigmas:
      gaussian_blurred = cv.GaussianBlur(output_image, (kernel_size, kernel_size),
                                        gaussian_sigma)

      # Border detection Method: Statistical range
      borders_detected = BorderDetectionStatisticalRange(
            padding=(5 // 2)
          ).process(gaussian_blurred)

      concatenated = np.concatenate((cv.cvtColor(output_image, cv.COLOR_GRAY2BGR),
                                    cv.cvtColor(gaussian_blurred, cv.COLOR_GRAY2BGR),
                                    cv.cvtColor(borders_detected, cv.COLOR_GRAY2BGR),
                                    calculate_watershed(borders_detected),
                                    ), axis=1)

      if write_images:
        cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
                  f'_statrange_border' \
                  f'_sigma{gaussian_sigma}' \
                  f'_kernelsize{kernel_size}' \
                  f'.jpg',
                  concatenated)
      else:
        if show_images:
          cv.imshow(f'{os.path.basename(filename)}' \
            f'_statrange_border' \
            f'_sigma{gaussian_sigma}' \
            f'_kernelsize{kernel_size}'
            f'.jpg',
            concatenated)
          cv.waitKey(0)
          cv.destroyAllWindows()

      # Border detection Method: Canny Border Detector

      # I don't know if moving this to after gaussian blur image calculation
      # changes things up so temporarily leave this to not make too many changes.
      gaussian_blurred = cv.normalize(gaussian_blurred, None, 0, 255,
                                      cv.NORM_MINMAX, cv.CV_8U)

      min_thresholds = [30, 51, 70]
      max_thresholds = [135, 150, 165]
      for min_threshold  in min_thresholds:
        for max_threshold in max_thresholds:
          borders_detected = cv.Canny(gaussian_blurred, min_threshold,
                                      max_threshold)
          borders_detected = cv.normalize(borders_detected, None, 0, 255,
                                          cv.NORM_MINMAX, cv.CV_8U)

          watershed_image = calculate_watershed(borders_detected)
          concatenated = np.concatenate((cv.cvtColor(output_image, cv.COLOR_GRAY2BGR),
                                         cv.cvtColor(gaussian_blurred, cv.COLOR_GRAY2BGR),
                                         cv.cvtColor(borders_detected, cv.COLOR_GRAY2BGR),
                                         watershed_image,
                                         ), axis=1)
          if write_images:
            cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
                       f'_statrange_border_sigma{gaussian_sigma}.jpg' \
                       f'canny_border' \
                       f'_sigma{gaussian_sigma}' \
                       f'_kernelsize{kernel_size}' \
                       f'_minThresh{min_threshold}' \
                       f'_maxThresh{max_threshold}' \
                       f'.jpg',
                       concatenated)
          else:
            if show_images:
              cv.imshow(f'{os.path.basename(filename)}' \
                        f'canny_border' \
                        f'_sigma{gaussian_sigma}' \
                        f'_kernelsize{kernel_size}' \
                        f'_minThresh{min_threshold}' \
                        f'_maxThresh{max_threshold}' \
                        f'.jpg',
                        concatenated)
              cv.waitKey(0)
              cv.destroyAllWindows()
